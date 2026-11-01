# The functional pipeline: nodes as local functionals, two stages, a linear decoder

This document describes a two-stage pipeline built from one primitive, reports what it
measures on the repository's CIFAR subset, and states which parts are settled and which are
not. It is meant to let someone else add a stage of their own and know whether it worked.

It is not a claim that the design is optimal, and it is not a comparison against published
CIFAR results: the split used here takes its test images from the same training batches, as
the `cifar_*` scripts do, so the numbers are internally comparable only.

---

## 1. What is fixed, and what is left open

**Fixed: the node.** A node is a parameterised function of a small patch,

    phi_j(p) = exp( -||p - c_j||^2 / h_j^2 ),     p = the k*k values of a k x k window

and it is admissible because `log phi_j` is **degree 2** in the patch entries, hence linear
in the natural parameters `(c_j / h_j^2, 1 / h_j^2)`. Two consequences follow, and they are
the reason this primitive was chosen:

* the geometry `(c_j, h_j)` can be recovered from a node's responses in closed form, so a
  fitted node does not require gradients;
* the condition is the same for every window size, so **one physical device whose response
  has this form can serve as a node** without changing any operator.

**Left open.** Everything else. In particular:

* **what the nodes are applied to** — the raw field, a residual, a normalised field, or the
  output of an earlier bank;
* **how many banks**, and how they hand a field to each other;
* **what is done with the response array** `phi_j(x)` — a (nodes x positions) array. This is
  not restricted to reducing it to a vector. Keeping it as a field, pooling it in regions,
  taking moments, or driving a further bank all belong here.

The two pipelines below are two particular choices. They are examples, not a template.

---

## 2. The two pipelines implemented here

**Pipeline A — a trend, then a residual.**

    m(x) = sum_j w_j phi_j(u)(x)        w fitted so that m approximates G_sigma * u
    r(x) = u(x) - lam * m(x)            a field again, same dimension as u

The target is a **blurred** image, not the image. Fitting `m` to `u` itself is kernel
interpolation: with 100 nodes centred on patches of the same images it reached `R^2 =
0.999` and left a residual carrying 0.6% of the variance. `sigma` is the knob that defines
what counts as trend, `lam` how much of it is removed, and `w` is fitted in closed form with
no labels.

**Pipeline B — features, then a linear decoder.**

    phi_k(r) -> positional mean -> ridge to +-1 one-vs-rest, lambda per class on validation

**Optimisation is per stage.** Stage A's weights are one least-squares solve; stage B's
centres are drawn from the residual's own patch distribution (or fitted by k-means, which
was measured to tie the draw); the readout is one ridge solve per class. There is no
cross-stage backpropagation, and adding data costs three closed-form solves.

---

## 3. Measured results

All on the repository's 5-class grayscale subset, `0.299/0.587/0.114`, 32x32.

| what was varied | result | conditions |
|---|---|---|
| node window 1x1 -> 3x3 | **+14.44** (CI [+10.79, +18.09]) | 5 seeds, 500 test images |
| subtract a local mean (sigma=2) | **+4.40** (CI [+1.89, +6.91]) | 5 seeds, 1000 test images |
| illumination slope, same change | **-2.55 -> +0.01 per 0.1** (change +2.56 [+0.79, +4.34]) | 5 seeds |
| training images 100 -> 300 per class | **+7.22** (CI [+5.46, +8.98]) | 10 seeds, 500 test images |
| greedy node placement, equal node count | **+3.34** (CI [+1.00, +5.68]) | 14 nodes, 5 seeds |
| nodes chosen greedily vs 100 drawn | -0.30 (CI [-1.51, +0.91]) | 54 vs 100 nodes |

Clean accuracy on this subset (100 training images per class, 100 nodes per bank):

| | accuracy |
|---|---|
| raw field | 46.54% |
| subtract a local mean, weights fitted | **50.94%** |
| same, weights fitted (learned bank A) | 51.34% |
| 300 training images per class | 53.74% |

**Baseline on the same protocol.** The repository's CNN, measured with the epoch chosen on a
100-image validation split rather than on the test set, with the same 400 training images
and the same 1000 test images: **51.14% (sd 2.24, 10 runs)**. The published 54.7% for the
same architecture is a maximum over 600 test evaluations with the weights restored from that
epoch; on these splits that protocol reports 53.26% against an honest 51.14%, a **+2.12
point** selection effect. The two pipelines are therefore indistinguishable on this
protocol (+0.20 points), and the previously published gap was a protocol artefact.

---

## 4. Adding your own stage

`experiments/fun_common.py` exposes the primitive and two hooks:

    transform_input(imgs) -> field      what the node bank is applied to
    use_field(phi)        -> anything   what is done with the (nodes x positions) array

Override either, keep the closed-form calibration, and the rest of the pipeline is
unchanged. The two `fun_01`/`fun_02` pipelines are two implementations; they are not
prescriptions.

**Before building a stage, three checks are worth running.** Each of them corresponds to a
way an earlier attempt in this repository failed:

1. **Which direction does it act on?** The positional mean in the readout already quotients
   the symmetric group on positions exactly (the pointwise model is bit-identical under a
   re-ordering of the point set, `2.220e-16`). A stage that acts on *positions* therefore has
   nothing to gain here; the mirror orbit measured -0.44 and rotation -0.16.
2. **Is there headroom?** `fun_common.probe_dg` measures `d_g = mean|f(u) - f(g.u)| / std(f)`
   on an already-fitted model with no labels. A small `d_g` means the model already ignores
   the transformation. Measured headroom: additive offset -2.84 points, gamma 1.3 -2.68,
   gain x1.15 -1.00, and gain-only is where the division stage came out at null.
3. **Does it compose?** An invariance survives a later stage only if the later stage
   preserves it. A mean over positions preserves any invariance of the field below it; a
   linear high-pass does **not** preserve a monotone (rank) invariance, and that composition
   cost -6.92 points.

---

## 5. Stages that were measured and did not pay

Reported so that a new attempt can skip them, and so the bounds are visible.

| stage | best result | bound |
|---|---|---|
| a second embedding (feature stack) | -5.04 (CI [-8.85, -1.23]) | resolved loss |
| a second embedding behind a **field** interface | +0.40 vs a zero-parameter fixed blur | CI [-0.96, +1.76], unresolved |
| six separators between the stages (sign split, gradient, Laplacian, energy, band) | +0.24 | <= 1.6 |
| a monotone (rank) quotient | -1.82 vs the raw field; +0.000 gamma slope, exactly | accuracy neutral |
| a multiplicative (division) quotient | -0.22 | <= 2.1 |
| higher-order readout (second scale, quadratic sketch) | -0.51 / +0.30 | <= 1.14 / <= 0.97 |

The second row is the informative one: a stack of embeddings loses partly because its second
layer's patch width is `n1*k*k`, which forces `n1*n2 <= budget/k^2`. Giving the two stages a
**field** interface removes that (the cost becomes additive), and even then a *fitted* first
stage was worth only +0.40 over a zero-parameter fixed blur.

The reason all of the stages above sit near zero is not that the ideas are wrong; it is that
400 training images against 100 columns leaves four images per column, and a stage placed
after the embedding competes for the same budget. The largest measured single lever is more
data (+7.22).

---

## 6. Evaluation rules used here

These are not stylistic. Each one corresponds to a result that was retracted when it was
missing.

1. **Equal capacity.** Compare arms at the same number of readout columns; a stage that
   multiplies channels gets correspondingly fewer nodes.
2. **Pair everything.** Same split *and* same node draw across arms. A paired test that only
   shares the split leaves the node draw in the comparison.
3. **Four states** for a paired difference: PASS / UNRESOLVED / NULL / FAIL. A positive mean
   whose interval contains zero is UNRESOLVED, not PASS and not FAIL.
4. **Test-set size goes in the conclusion.** 150 test images means one image is 0.67 points
   and a single cell carries about +-4 points of binomial error; at that size a resolved
   null can turn into a resolved loss when the test set grows.
5. **Degenerate cells are not scored.** With `sigma = 0` the `fixed` arm makes the residual
   identically zero; it was once recorded as 20.00% (chance, from argmax over NaN) and
   produced two conclusions that had to be withdrawn.
6. **Self-checks where a transformation has an exact property.** A rank-transformed feature
   must show a gamma slope of exactly zero; if it does not, the transform order is wrong and
   no accuracy number downstream is readable.
7. **One seed flips signs in both directions.** Not "single seeds are optimistic": the
   second-scale arm went from -2.20 at one seed to +0.34 at five, and the division arm went
   from +1.60 at one seed to -0.22 at five.

---

## 7. Retractions

Listed because they were once reported and are still findable.

| reported | actual | why |
|---|---|---|
| "17 nodes are worth 100" | 14 vs 100 nodes: -2.62, resolved | 3 seeds, 150 test images |
| "greedy placement is 2.62 worse than sampling" | invalid | compared 14 greedy nodes against 100 drawn |
| "the orbit flattens the illumination curve by 33%" | +0.00 | the two arms had not shared a node draw |
| "the mirror orbit is worth +4.45" | -0.44 | measured while the validation split was a single class |
| "the division stage adds a second class-1 member" | -0.22 | 1 seed; five seeds made it null |
| "the designed lattice covers enough" | coverage 0.714 vs 0.919 | the lattice lay in a two-dimensional slice of a nine-dimensional space |

---

## 8. Reproduce

```bash
python experiments/fun_01_detrend.py --seeds 5 --n-test 200 --sigma 2
python experiments/fun_02_baseline_remeasure.py --data-seeds 5 --init-seeds 2
```

Expected: `fun_01` raw 46.54%, fixed 50.94%, learned 51.34%, illumination slope -2.55 -> +0.01;
`fun_02` honest 51.14% (sd 2.24), peak_test 53.26%, selection effect +2.12.

---

## 9. Boundaries

* 5 classes, 32x32 grayscale, 100-300 training images per class, no augmentation.
* **Test images come from the training batches**, as in the `cifar_*` scripts. The numbers
  are internally comparable and nothing more.
* The published 36.0% for `cifar_02` is a single seeded draw with no interval; 54.7% for the
  CNN includes a test-set selection effect of +2.12 points.
* Accuracy is not where this differs from a small CNN. Parameter count is not either: the
  CNN has ~2600, this pipeline ~1500. What differs is that every stage here is a closed-form
  solve and the only nonlinear primitive is one whose admission condition is stated on a
  device's response.
