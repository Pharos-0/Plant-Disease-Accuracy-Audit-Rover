# GdScore — eligibility and reproduction record

**Phase 7, step 1. Written before any benchmark use**, as required: a candidate estimator
is classified eligible/ineligible and its reproduction fidelity established *before* it is
allowed into the comparison.

**The method.** Xie, Odonnat, Feofanov, Redko, Zhang & An, *"Leveraging Gradients for
Unsupervised Accuracy Estimation under Distribution Shift"*, **TMLR 2025**
(arXiv:2401.08909). Official implementation obtained:
`github.com/Renchunzi-Xie/GdScore`, cloned to `C:\ds\_refrepos\GdScore`.

Everything below is read from the official `algs/gdscore.py` and verified numerically
against our own models — not inferred from the abstract.

---

## 1. Verdict

| Question | Answer |
|---|---|
| Eligible for our protocol (frozen classifier, unlabelled target)? | **YES** |
| Reproducible from our existing cached logits + features? | **NO for 3 of 4 backbones** — see §4 |
| Requires autograd / a backward pass? | **No** — the gradient is closed-form (§3) |
| Requires a fresh forward pass over target images? | **Yes, in `train()` mode** (§4) |
| Ready to enter the benchmark unchanged? | **No** — a protocol obstacle must be resolved first (§6) |

**Recommendation: eligible, implementable, but the session-level protocol needs a decision
before it can join the N-sweep.**

---

## 2. What the official code actually computes

```python
self.base_model.train()                    # NOTE: train mode, not eval
...
outputs = self.base_model(inputs)
p, pseudo_labels = softmax(outputs).max(1)
pseudo_labels[p < t] = random labels        # t = 0.5
loss = CrossEntropyLoss(outputs, pseudo_labels)
loss.backward()
weight = self.base_model.fc.weight.grad
score = torch.norm(weight, p=self.args["norm_type"])
```
averaged over batches. Hyperparameters from `bash/gdscore.sh`:
`threshold=0.5`, `norm_type=0.3`, `batch_size=128`, `lr=0.001`, `seed=1`.

### Four details that are easy to get wrong

1. **`norm_type = 0.3`, not the Frobenius norm.** This is a *fractional* entrywise
   p-"norm" (it is not a norm — it violates the triangle inequality for p<1). The scale
   difference is enormous: on one test gradient we measured

   | p | value |
   |---|---:|
   | 0.3 | 1.37e10 |
   | 1 | 25.47 |
   | 2 (Frobenius) | 0.41 |

   Implementing this with the Frobenius norm would be a different estimator.

2. **`optimizer.step()` is never called.** An SGD optimizer is constructed and `lr` is
   passed, but only `zero_grad()` is used; the gradient is read at the *original* weights.
   Despite "one-step gradient" in the paper's framing, **the released code takes no step.**
   This is the same class of discrepancy as the ATC release fitting a `TempScaling`
   calibrator and then not applying it, which we already document. We would follow the
   released code and record the divergence, as we did for ATC.

3. **Low-confidence samples get a *random* pseudo-label** (`p < 0.5`). The score is
   therefore stochastic and requires seed control — material here, because our field
   confidences are much lower than the CIFAR/ImageNet settings the method was tuned on.

4. **`model.train()` mutates BatchNorm running buffers.** Any implementation must
   deep-copy the model or restore its buffers afterwards, or the "frozen" classifier
   silently drifts for every later evaluation.

---

## 3. The gradient is closed-form — no autograd needed

For a linear head, the gradient of the batch-mean cross-entropy with respect to the head
weight is

```
grad_W = (1/B) · Rᵀ H ,      R = softmax(logits) − onehot(pseudo_labels) ,  H = penultimate features
```

**Verified against autograd:** maximum absolute difference **1.9e-09** (B=128, D=2048,
K=3). So GdScore needs no backward pass at all — only `p` and `h`.

*(Correction to my earlier claim: I previously suggested the per-sample identity
‖(p−y)hᵀ‖_F = ‖p−y‖·‖h‖ would make this trivial. That identity is true but **not
applicable** — GdScore takes the norm of a **sum over the batch**, and the norm of a sum
of outer products does not factorise. The matrix form above is the correct route.)*

---

## 4. Can it be reproduced from our existing cache? **Mostly no.**

Our Stage-2 cache stores `eval()`-mode logits and features. GdScore runs the forward pass
in `train()` mode, where BatchNorm uses **batch statistics** instead of running statistics.
Measured on our own trained models, one batch of 128:

| backbone | BatchNorm | max abs Δlogit | max abs Δfeature | prediction agreement |
|---|---|---:|---:|---:|
| resnet50 | yes | 4.40 | 3.47 | **0.461** |
| efficientnet_b0 | yes | 8.96 | 2.01 | **0.453** |
| mobilenetv3_large | yes | 6.09 | 1.64 | **0.570** |
| **vit_small_patch16** | **no** | **0.0000** | **0.0000** | **1.000** |

**Train and eval mode disagree on more than half the predictions** for every BatchNorm
backbone. The cached activations are therefore the wrong inputs for a faithful GdScore.

**ViT-S/16 is bit-identical** — LayerNorm has no batch dependence and dropout is inactive
— so for that backbone alone the cache is exactly sufficient.

**Consequence:** a faithful implementation needs a **fresh `train()`-mode forward pass**
over the target images for the three BatchNorm backbones. That is an inference run
comparable in cost to our existing caching stage (23.7 min for 120 model×arm pairs), with
**no retraining and no autograd**.

---

## 5. Eligibility against our protocol

| Requirement | GdScore | Verdict |
|---|---|---|
| Classifier weights frozen | weights never updated; only BN buffers drift, and that is restorable | **PASS** |
| No target labels at estimation | uses its own pseudo-labels only | **PASS** |
| Dataset/session-level, not per-image | score is a per-batch aggregate | **PASS** |
| Source labels permitted | not required at all | **PASS** |
| Deterministic given a seed | yes, once the random-label seed is fixed | **PASS with seed control** |

**Eligible.** Unlike AETTA (assumes a test-time-*adapting* model) and AutoEval (meta-training
needs labelled sample sets), GdScore fits the frozen-classifier deployment setting directly.

---

## 6. The obstacle that must be resolved before it joins the benchmark

**GdScore's score is strongly batch-size dependent.** Measured on identical data (512
samples), varying only the batching:

| batch size | mean score |
|---:|---:|
| 25 | 2.20e10 |
| 64 | 1.72e10 |
| 128 | 1.32e10 |
| 512 | 6.87e09 |

A **3.2× swing** from batching alone. Two mechanisms contribute: the `1/B` factor in the
gradient, and — for BatchNorm backbones — the batch statistics themselves.

This collides directly with our N-sweep, where sessions run from 25 to 400 images. **Naively
scoring a session as one batch would confound session size with the score's own batch-size
dependence**, and the resulting N-curve would be an artefact.

Three options, none yet chosen:

| option | description | cost |
|---|---|---|
| **A** | Fix `batch_size = 128` as the paper does; evaluate only on the full pool and on sessions with N ≥ 128 | loses N = 25, 50, 100 from the sweep |
| **B** | Score each session as a single batch, and report the batch-size dependence as a property of the method | keeps the sweep but the N-curve is partly an artefact |
| **C** | Evaluate GdScore only on the **full unlabelled pool** (the RQ5 comparison), not in the N-sweep | cleanest; answers "is it a better estimator" without touching the sample-efficiency question |

**Recommendation: option C first, then A if the result warrants it.** Option C answers the
question that actually motivated Phase 7 — *does a stronger, later estimator change our
conclusions?* — and it does so on exactly the comparison RQ5 already established, where
every baseline is scored on all 1,616 unlabelled crops. It avoids inventing a protocol for
a method whose author-specified batch size exceeds most of our session sizes.

Additionally, for BatchNorm backbones a session of 25 images would compute BN statistics
from 25 samples, which is unstable regardless of the score definition. That is an
independent reason not to push GdScore into the small-N regime.

---

## 7. Planned reproduction level

Following `docs/estimator_reproducibility.md` conventions:

- **Target: L3** — reproduce the official `algs/gdscore.py` computation exactly, verified by
  a test that reimplements the reference path (train-mode forward → closed-form gradient →
  `torch.norm(p=0.3)`) and compares against an autograd implementation, as §3 already does
  for the gradient itself.
- It will be recorded as **our implementation of the official algorithm**, with the
  `optimizer.step()` divergence documented, not as the authors' code executed verbatim
  (their harness is tied to CIFAR/ImageNet loaders and their own checkpoints).

---

## 8. What has NOT been decided

- Whether GdScore enters the thesis benchmark. That depends on the result, and on the
  protocol choice in §6.
- Nothing has been run beyond the verification above. No estimator has been added to
  `src/estimators/`, and no existing result has been touched.
