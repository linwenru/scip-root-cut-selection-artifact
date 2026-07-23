# Online causal harness

The first online stage is a hard validity check, not an ML experiment. It asks
whether a high-priority PySCIPOpt cut selector can observe a selection callback,
return `SCIP_DIDNOTFIND`, and leave SCIP's native complete B&B trajectory
structurally unchanged.

## No-op parity gate

Each arm and seed runs in a fresh Python process:

- `native`: SCIP with no custom cut selector.
- `noop`: a high-priority selector counts callback activity, does not reorder the
  candidate array, and returns `SCIP_DIDNOTFIND` so SCIP may try its next selector.

Both arms use the same three SCIP randomization seeds and one solver thread. A
pair passes only when status, objective sense, primal and dual bounds, gap,
processed and total nodes, LP iterations, LP count, and cuts applied match. The
no-op callback must also have run at least once.

Solving time and primal-dual integral are recorded but do not gate no-op parity.
Both depend on wall-clock time, so the Python callback itself can perturb them
without changing SCIP's structural decisions. They become solve-level outcomes
in later randomized treatment experiments.

Run a pilot with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m \
  scip_cut_trace_v2.causal_harness noop-parity \
  /path/to/instance.mps.gz \
  --seeds 0 1 2 \
  --time-limit 300
```

The command exits with status 2 when any pair fails. Raw arm results are written
under `experiments/`, while the compact gate manifest is written to
`data/manifests/causal_noop_parity_v1.json`.

The initial complete-B&B pilot on `neos-3381206-awhea` passed for seeds 0, 1,
and 2. The no-op selector was exercised 37, 19, and 47 times respectively,
while all gating structural outcomes matched native SCIP exactly.

## Native hybrid bridge

The treatment must preserve SCIP's native selected-cut count. A narrow Cython
bridge therefore calls the public `SCIPselectCutsHybrid()` function with the
already initialized hybrid plugin data and returns the native ordering and
selected count. The bridge is guarded to SCIP 10.0.2 because the hybrid plugin
data layout is not a public ABI.

`direct-hybrid` passed the same three-seed structural parity gate. For each seed,
the direct bridge reproduced native status, bounds, nodes, LP iterations, LP
count, and applied cuts exactly. This is the required validity check before any
ordering perturbation.

## Boundary-swap treatment

The first causal treatment operates at the first eligible root callback in each
SCIP run. It calls native hybrid, keeps its selected count, and swaps the last
selected row with the first unselected row. Later callbacks in the same run
delegate to native SCIP. Restarted runs receive a fresh budget of at most one
intervention.

The three-seed pilot on `neos-3381206-awhea` was valid but unstable: node counts
changed `132 -> 1`, `21 -> 40`, and `280 -> 1`. This proves that a single root
selection change can alter the complete B&B trajectory, but not that the fixed
action is safe.

A 10-instance training-only suite reinforced that distinction. All instances
were eligible; among comparable instance means, LP iterations had 2 wins, 1 tie,
and 7 losses. More importantly, `cbs-cta` seed 1 changed from native optimal in
18.81 seconds to treatment timeout at 120 seconds with a large remaining gap.
The fixed boundary-swap policy is therefore rejected. Metric aggregates exclude
that noncomparable censored pair and must not be read as a safety result; the
suite-level safety flag is false.

## Stage boundary

No treatment policy is valid until both structural parity gates pass. The next
stage is a small, predeclared action-library oracle experiment. It will test
whether any deterministic root action has repeatable solve-level gains on
training instances. No ML policy is reopened unless an action survives new-seed
validation without catastrophic losses.

## Predeclared action-library oracle

The first action library is fixed before its outcomes are inspected:

- `boundary-swap`: replace the last native-selected cut with the first
  native-unselected cut.
- `boundary-swap-2`: replace the last native-selected cut with the second
  native-unselected cut.
- `efficacy-promote`: replace the least efficacious native-selected cut with the
  most efficacious native-unselected cut, but only for a strict efficacy gain.

Every action preserves the native selected-cut count and may intervene at most
once per SCIP run, at an eligible root callback. Each action and native SCIP run
in fresh processes with identical seeds and limits. The primary oracle metric is
LP iterations; native wins ties. Nodes, solve time, and primal-dual integral are
still reported. A pair is valid only when both arms complete and have matching
status and final bounds.

The oracle chooses the best valid action separately after seeing each
instance-seed outcome. It is deliberately optimistic and is only a causal
upper bound. It cannot be deployed, and its fallback to native does not erase an
unsafe action's timeout or mismatched final result.

Run a training-only pilot with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m \
  scip_cut_trace_v2.causal_harness action-oracle-suite \
  /path/to/first.mps.gz /path/to/second.mps.gz \
  --seeds 0 1 2 \
  --time-limit 120 \
  --output-dir experiments/action_oracle_v1 \
  --manifest data/manifests/causal_action_oracle_v1.json
```

The initial pilot used five training instances and three seeds, for 15 shared
native baselines and 45 treatment arms. `boundary-swap` completed safely but had
one instance win and four losses, with an instance-equal mean relative LP
iteration saving of -10.35%. `boundary-swap-2` was also safe and reached 2 wins
and 3 losses, with +1.81% mean relative saving. `efficacy-promote` reached 2 wins
and 3 losses among comparable instance means, but was unsafe: on
`neos-1171448` seed 1 it changed a 5.08-second native optimum into a 120-second
timeout.

The post-hoc per-seed oracle selected a treatment in 10 of 15 pairs and native
in 5. It reported a +9.93% instance-equal mean relative LP-iteration saving and
an oracle win on all five instances. That is only the optimistic upper bound.

As a stability check, each held-out seed was evaluated using the action with the
largest positive mean saving on the other two seeds. Actions with an unsafe
training seed were excluded, and native won ties. This leave-one-seed-out rule
failed: its instance-equal mean relative saving was -22.49%, with 1 instance
win, 4 losses, and one unsafe held-out seed. At seed level it had 2 wins, 6 ties,
and 6 losses among evaluable outcomes.

The action library therefore demonstrates causal opportunity but not a stable
selection rule. No fixed action or seed-history selector advances to deployment,
and the post-hoc oracle must not be used as an ML label without collecting
substantially more active, state-rich causal observations.

## First-run-only identification pilot

The per-run policy is a valid deployment-level treatment, but a solve with SCIP
restarts can contain several interventions. Its final outcome cannot be cleanly
assigned to one root decision. Active causal data collection therefore supports
an additional `first-run-only` scope: it consumes one decision at the first
eligible root callback of run 1 and delegates every later restart to native
SCIP. This scope identifies one action per complete solve; it does not change the
eventual deployment budget of at most one action per run.

Each first-run treatment records a leakage-safe pre-intervention context. It
contains solver, node, LP, and bound state plus the native hybrid candidate
ordering and per-row structural features. It excludes solving time, final solve
status, final bounds, and every post-action outcome. Audit names are retained but
are not model features. The raw context is stored only in the process-isolated
arm JSON; the Git manifest stores its SHA-256 and compact selector metadata.

The same five-instance, three-seed pilot was repeated in this scope. All 15
instance-seed pairs had identical context hashes across the three action arms;
all 45 treatment arms recorded exactly one context and at most one intervention.
The causal result did not become stable:

- `boundary-swap`: safe, 1 instance win and 4 losses, -6.26% instance-equal
  mean relative LP-iteration saving.
- `boundary-swap-2`: safe, 2 wins and 3 losses, +1.55% mean relative saving.
- `efficacy-promote`: 2 wins and 3 losses among comparable means, -13.10% mean
  relative saving, plus the same `neos-1171448` seed 1 timeout.
- post-hoc oracle: +10.85% mean relative saving and 5 instance wins.
- leave-one-seed-out selector: -31.19% mean relative saving, 1 instance win,
  4 losses, and one unsafe held-out seed.

The instability is therefore caused by the first root action itself, not only
by repeated interventions after restarts. These 15 contexts validate the causal
data contract but are far too few to train an action or abstention model.
