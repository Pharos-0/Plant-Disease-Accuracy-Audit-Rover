"""Autonomous selective pesticide sprayer: FSM control, sensor fusion, spray policy.

This module is the control core of the sprayer. It is written to be run both on
the rover and in simulation, with no branch anywhere that asks which it is: the
FSM consumes a fused sensor reading and emits an actuation command, and whether
those readings come from real GPIO or from the digital twin is the caller's
problem, not the controller's.

DESIGN: WHY A FINITE STATE MACHINE
-----------------------------------
A sprayer carries a tank of pesticide over a field with people and livestock
potentially nearby. The property that matters is not "does it spray well" but
"is there any reachable condition in which it sprays when it should not". A
behaviour tree or a policy network cannot answer that question by inspection; a
finite state machine with an explicit transition table can. Every state here
declares whether spraying is permissible in it, the permission is checked in one
place, and `unreachable_spray_states()` enumerates the states from which a spray
command can never be issued so the claim is testable rather than asserted.

The FSM is also what makes a fault safe. There is exactly one terminal state,
FAULT_SAFE_STOP, it is reachable from every other state, and it is the only
state with no outgoing transition except to DOCKED. Anything the controller does
not understand lands there with the pump off.

SENSOR FUSION
-------------
Four inputs, deliberately of different physical kinds so that no single failure
blinds the machine:

  * ultrasonic range    - obstacle, the hard safety stop
  * infrared row sensor - lateral position in the row, for navigation
  * vision              - canopy presence per nozzle zone, and disease class
  * resource monitor    - tank level and battery state of charge

Fusion is not averaging. Each input has authority over a different decision and
a different failure response, which is why `FusedReading.validate()` reports
*which* input is stale rather than returning a single confidence number: a stale
ultrasonic reading must stop the platform, while a stale disease classification
need only disable selective spraying and fall back to not spraying at all.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not claim any field performance. It computes decisions. What those
decisions cost in pesticide, and what they miss, is measured in
`scripts/run_sprayer_experiment.py` against real imagery, and is reported there
as a simulation result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable

# --------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------


class State(Enum):
    """Controller states. `SPRAY_OK` below is the single source of truth for
    whether the pump may be energised in each one."""

    IDLE = auto()             # powered, stationary, pump locked out
    TRAVERSING = auto()       # moving along a row, looking for targets
    TARGET_LOCKED = auto()    # canopy detected ahead, preparing nozzles
    SPRAYING = auto()         # pump energised on at least one nozzle
    OBSTACLE_STOP = auto()    # ultrasonic triggered; motion and pump halted
    LOW_RESOURCE = auto()     # tank or battery below reserve; heading home
    RETURNING = auto()        # autonomous return to the dock
    DOCKED = auto()           # arrived; everything released
    FAULT_SAFE_STOP = auto()  # terminal; pump off, motors off


#: The single place that decides whether spraying is permitted. Every spray
#: command is checked against this, so a new state cannot silently become
#: sprayable by someone forgetting to update a condition elsewhere.
SPRAY_OK: dict[State, bool] = {
    State.IDLE: False,
    State.TRAVERSING: False,
    State.TARGET_LOCKED: False,
    State.SPRAYING: True,
    State.OBSTACLE_STOP: False,
    State.LOW_RESOURCE: False,
    State.RETURNING: False,
    State.DOCKED: False,
    State.FAULT_SAFE_STOP: False,
}

#: Legal transitions. Anything not listed is a programming error and raises.
TRANSITIONS: dict[State, set[State]] = {
    State.IDLE: {State.TRAVERSING, State.FAULT_SAFE_STOP},
    State.TRAVERSING: {State.TARGET_LOCKED, State.OBSTACLE_STOP,
                       State.LOW_RESOURCE, State.RETURNING,
                       State.FAULT_SAFE_STOP, State.TRAVERSING},
    State.TARGET_LOCKED: {State.SPRAYING, State.TRAVERSING, State.OBSTACLE_STOP,
                          State.LOW_RESOURCE, State.FAULT_SAFE_STOP},
    State.SPRAYING: {State.TRAVERSING, State.OBSTACLE_STOP, State.LOW_RESOURCE,
                     State.FAULT_SAFE_STOP, State.SPRAYING},
    State.OBSTACLE_STOP: {State.TRAVERSING, State.RETURNING,
                          State.FAULT_SAFE_STOP},
    State.LOW_RESOURCE: {State.RETURNING, State.FAULT_SAFE_STOP},
    State.RETURNING: {State.DOCKED, State.OBSTACLE_STOP, State.FAULT_SAFE_STOP,
                      State.RETURNING},
    State.DOCKED: {State.IDLE, State.FAULT_SAFE_STOP},
    State.FAULT_SAFE_STOP: {State.DOCKED},     # only a human-reset path
}


def unreachable_spray_states() -> set[State]:
    """States from which a spray command can never be issued.

    Exposed so the safety claim is a test rather than a sentence in a report.
    """
    return {s for s, ok in SPRAY_OK.items() if not ok}


# --------------------------------------------------------------------------
# Sensor fusion
# --------------------------------------------------------------------------

#: Stop distance, centimetres. Below this the platform halts regardless of state.
OBSTACLE_STOP_CM = 25.0
#: Tank fraction at which the rover must break off and return.
TANK_RESERVE = 0.08
#: Battery state of charge at which it must break off and return.
BATTERY_RESERVE = 0.15
#: A reading older than this many control cycles is stale.
STALE_CYCLES = 3


@dataclass
class FusedReading:
    """One fused observation. Ages are in control cycles since last update."""

    range_cm: float
    range_age: int
    row_offset: float            # IR: -1 hard left .. +1 hard right, 0 centred
    row_age: int
    canopy: tuple[float, ...]    # vegetation fraction per nozzle zone
    disease_confidence: float    # session-level, label-free
    vision_age: int
    tank_fraction: float
    battery_fraction: float
    resource_age: int
    #: Dock beacon. Without an arrival signal RETURNING could never complete, and
    #: the autonomous-return capability would be a claim with no test behind it.
    at_dock: bool = False
    #: Per-nozzle probability that the zone holds the CROP, from a trained
    #: detector. None => fall back to the vegetation fraction in `canopy`, which
    #: cannot tell a tomato from a weed.
    crop_prob: tuple[float, ...] | None = None
    #: Per-nozzle disease confidence. None => use the single `disease_confidence`.
    disease_by_zone: tuple[float, ...] | None = None

    def validate(self) -> list[str]:
        """Return the names of stale inputs. Deliberately not a single score:
        each input has a different failure response, so the caller needs to know
        which one is missing, not how bad things are on average."""
        bad = []
        if self.range_age > STALE_CYCLES:
            bad.append("ultrasonic")
        if self.row_age > STALE_CYCLES:
            bad.append("infrared")
        if self.vision_age > STALE_CYCLES:
            bad.append("vision")
        if self.resource_age > STALE_CYCLES:
            bad.append("resource")
        return bad


@dataclass
class Command:
    """What the controller wants done this cycle."""

    state: State
    drive: str                       # "forward" | "stop" | "return"
    nozzles: tuple[bool, ...]        # one flag per nozzle
    steer: float                     # -1 .. +1, from the IR row sensor
    reason: str

    @property
    def spraying(self) -> bool:
        return any(self.nozzles)


# --------------------------------------------------------------------------
# Spray policy
# --------------------------------------------------------------------------

#: Vegetation fraction in a nozzle zone above which there is something to spray.
CANOPY_THRESHOLD = 0.15
#: Crop-detector probability above which a zone is treated as holding the crop.
CROP_THRESHOLD = 0.5


class SprayPolicy(Enum):
    """The three policies compared in the experiment."""

    BLANKET = "blanket"          # every nozzle, every cycle: the manual baseline
    VISION_ONLY = "vision_only"  # fire where canopy is detected
    SELECTIVE = "selective"      # canopy AND disease AND the reliability gate


def nozzle_decision(policy: SprayPolicy, canopy: tuple[float, ...],
                    disease_conf: float, gate_open: bool,
                    crop_prob: tuple[float, ...] | None = None,
                    disease_by_zone: tuple[float, ...] | None = None,
                    ) -> tuple[bool, ...]:
    """Which nozzles should fire, before any state-machine veto.

    The gate is deliberately *outside* the vision decision. A classifier that is
    confident and wrong is the failure this project exists to document, so the
    selective policy refuses to act on a disease classification the session-level
    reliability check has not endorsed. When the gate is shut it falls back to
    VISION_ONLY rather than to BLANKET: spraying the canopy it can see is a
    defensible default, spraying everything is not.
    """
    n = len(canopy)
    if policy is SprayPolicy.BLANKET:
        return tuple([True] * n)

    has_canopy = tuple(c >= CANOPY_THRESHOLD for c in canopy)
    if policy is SprayPolicy.VISION_ONLY:
        return has_canopy

    # The selective policy targets the CROP when a detector is available, and
    # the vegetation mask only when it is not.
    target = (tuple(p >= CROP_THRESHOLD for p in crop_prob)
              if crop_prob is not None else has_canopy)
    if not gate_open:
        return target                          # degrade to crop-only, never blanket
    dis = disease_by_zone if disease_by_zone is not None else (disease_conf,) * n
    return tuple(t_ and d >= 0.5 for t_, d in zip(target, dis))


# --------------------------------------------------------------------------
# The controller
# --------------------------------------------------------------------------


@dataclass
class SprayerController:
    """Deterministic FSM over fused sensor readings.

    `step()` is total: every input produces a command, and any condition the
    machine cannot interpret produces FAULT_SAFE_STOP with the pump off.
    """

    policy: SprayPolicy = SprayPolicy.SELECTIVE
    n_nozzles: int = 4
    state: State = State.IDLE
    history: list[State] = field(default_factory=list)
    _faulted: bool = field(default=False, init=False)

    # -- transition ---------------------------------------------------------
    def _go(self, nxt: State) -> None:
        if nxt not in TRANSITIONS[self.state]:
            raise ValueError(f"illegal transition {self.state.name} -> {nxt.name}")
        self.state = nxt
        self.history.append(nxt)

    def _fault(self, why: str) -> Command:
        if self.state is not State.FAULT_SAFE_STOP:
            self._go(State.FAULT_SAFE_STOP)
        self._faulted = True
        return Command(self.state, "stop", (False,) * self.n_nozzles, 0.0, why)

    # -- main ---------------------------------------------------------------
    def step(self, r: FusedReading, *, gate_open: bool) -> Command:
        off = (False,) * self.n_nozzles

        if self._faulted:
            return Command(State.FAULT_SAFE_STOP, "stop", off, 0.0,
                           "latched fault; requires a human reset")

        if len(r.canopy) != self.n_nozzles:
            return self._fault(f"vision reported {len(r.canopy)} zones, "
                               f"expected {self.n_nozzles}")

        stale = r.validate()
        # Ultrasonic is the one input whose loss is itself an emergency: without
        # it the platform cannot know whether it is about to hit something.
        if "ultrasonic" in stale:
            return self._fault("ultrasonic reading stale; cannot guarantee clearance")
        if "resource" in stale:
            return self._fault("resource telemetry stale; tank level unknown")

        if self.state is State.IDLE:
            self._go(State.TRAVERSING)
            return Command(self.state, "forward", off, 0.0, "starting traverse")

        if self.state is State.DOCKED:
            return Command(self.state, "stop", off, 0.0, "docked")

        # -- hard safety, in priority order --------------------------------
        if r.range_cm < OBSTACLE_STOP_CM:
            if self.state is not State.OBSTACLE_STOP:
                self._go(State.OBSTACLE_STOP)
            return Command(self.state, "stop", off, 0.0,
                           f"obstacle at {r.range_cm:.0f} cm")

        if self.state is State.OBSTACLE_STOP:
            self._go(State.TRAVERSING)
            return Command(self.state, "forward", off, r.row_offset, "obstacle cleared")

        # -- resources ------------------------------------------------------
        low = r.tank_fraction <= TANK_RESERVE or r.battery_fraction <= BATTERY_RESERVE
        if low and not (self.state is State.RETURNING and r.at_dock):
            if self.state not in (State.LOW_RESOURCE, State.RETURNING):
                self._go(State.LOW_RESOURCE)
            if self.state is State.LOW_RESOURCE:
                self._go(State.RETURNING)
            return Command(self.state, "return", off, r.row_offset,
                           f"reserve reached: tank {r.tank_fraction:.0%}, "
                           f"battery {r.battery_fraction:.0%}")

        if self.state is State.RETURNING:
            if r.at_dock:
                self._go(State.DOCKED)
                return Command(self.state, "stop", off, 0.0, "arrived at dock")
            return Command(self.state, "return", off, r.row_offset, "returning to dock")

        # -- spraying -------------------------------------------------------
        for name, vec in (("crop_prob", r.crop_prob),
                          ("disease_by_zone", r.disease_by_zone)):
            if vec is not None and len(vec) != self.n_nozzles:
                return self._fault(f"{name} has {len(vec)} zones, "
                                   f"expected {self.n_nozzles}")
        want = nozzle_decision(self.policy, r.canopy, r.disease_confidence, gate_open,
                               r.crop_prob, r.disease_by_zone)

        # Vision loss disables selective spraying rather than defaulting to on.
        if "vision" in stale:
            want = off

        if any(want):
            if self.state is State.TRAVERSING:
                self._go(State.TARGET_LOCKED)
            if self.state is State.TARGET_LOCKED:
                self._go(State.SPRAYING)
            if not SPRAY_OK[self.state]:
                return self._fault(f"spray requested in {self.state.name}")
            steer = 0.0 if "infrared" in stale else r.row_offset
            return Command(self.state, "forward", want, steer,
                           f"{sum(want)}/{self.n_nozzles} nozzles on canopy")

        if self.state in (State.SPRAYING, State.TARGET_LOCKED):
            self._go(State.TRAVERSING)
        steer = 0.0 if "infrared" in stale else r.row_offset
        return Command(self.state, "forward", off, steer, "no target")
