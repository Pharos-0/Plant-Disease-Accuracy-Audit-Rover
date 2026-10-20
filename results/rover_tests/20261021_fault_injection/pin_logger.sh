#!/bin/bash
# Log the motor-driver pins (BCM 17, 22, 23, 27) ten times a second for $1 seconds.
# One line per sample: "<epoch seconds> 17=hi 22=hi 23=lo 27=lo".
end=$(( $(date +%s) + ${1:-25} ))
while [ "$(date +%s)" -lt "$end" ]; do
  echo "$(date +%s.%N) $(pinctrl get 17,22,23,27 | sed -E 's/^([0-9]+):.*\| (lo|hi).*/\1=\2/' | tr '\n' ' ')"
  sleep 0.1
done
