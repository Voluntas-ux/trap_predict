# Source context

The verification report refers throughout to two earlier documents that are not bundled here: a design note for the single-round independence diagnostic, and a work plan for the wider target-prediction project. This file records what they specify, so the repository stands on its own.

## The object being modelled

Olympic Trap is shot by a squad of six athletes across five stations. Each athlete visits each station five times, and at each station receives two left targets, two right targets and one straight, in an order the athlete does not know. A round is therefore a 6 x 5 x 5 array of direction labels, 150 targets in total, arranged as 30 rows, each row a permutation of the multiset {L, L, S, R, R}. There are 30 such permutations.

Athletes shoot in rotation, so the round has a firing order as well as an array structure: the athlete on station 1 shoots, then station 2, and so on, with athletes moving one station to the right after each target. Under the standard six-athlete rotation, shot n (counting from zero) is athlete n mod 6 at station n mod 5 on visit floor(n / 30).

## What was already settled before this work

- The exact single-station urn chain: state space, transition probabilities, conditional entropy, and optimal-guess accuracy. Because the allocation at a station is two left, two right and one straight, an athlete who counts what they have already received at that station can do better than chance. The optimal counting accuracy over a full round is 60.67 per cent, verified by exhaustive enumeration and by Monte Carlo. Blind guessing is 33.3 per cent; always guessing one side is 40 per cent.
- The factorisation result: if the five station urns are independent and the scheduler is a plain round robin, no information passes between stations, and nothing can be inferred about an athlete's next target from any other athlete's targets.
- Closed-form angular kinematics for the first fraction of a second of flight, under stated launch assumptions.

## What was unvalidated, and gating everything else

- Whether the release controller actually draws uniformly over the 30 orderings.
- Whether stations and athletes are actually independent.
- Whether the launch speed and station distance used in the kinematics match any specific range.

The work plan is explicit that the independence question blocks the rest: if independence fails, the factorisation result, the 60.7 per cent ceiling, and the claim that nothing can be inferred before the sixth shot all have to be rebuilt before any downstream vision or fusion work has a meaningful prior.

## The work plan, in outline

- **W1 Validate the randomisation assumptions.** Data collection protocol; a chi-square test of uniformity over the 30 orderings (W1.2); tests for local constraints such as runs and serial correlation (W1.3); a test of independence across stations and athletes (W1.4); and a consequence branch specifying what must be rebuilt if independence is rejected (W1.5).
- **W2 Scheme identification and per-day range calibration.** Digitise the machine setting tables; survey the range geometry; estimate launch parameters per machine per day from video.
- **W3 Ballistic forward model** for a spinning disc with lift, drag and spin decay, fitted to observed tracks.
- **W4 Vision pipeline.** High-frame-rate global-shutter capture, detection against the pit roof line, and measured end-to-end latency.
- **W5 Single-frame appearance classifier.** Emergence x-coordinate and silhouette geometry as explicit features before any learned model, with occlusion analysis to catch a model reading range artefacts instead of the target.
- **W6 Fusion and stopping rule.** Prior from the urn state, per-frame likelihoods from a matched-filter bank, and a sequential probability ratio test for early stopping.
- **W7 Evaluation protocol and leakage controls.** Baselines at 33.3, 40.0 and 60.7 per cent; holdout by range, by scheme and by session rather than at random; pre-registration; and standing checks including label shuffling and a first-two-frames ablation.
- **W8 Extensions** beyond the single round: the five-round competition, scheme rotation across days, the finals format, and squadmate observation.
- **W9 Deliverables and reproducibility.** Every table regenerable by a single script, empty cells rather than estimates for unmeasured quantities, fixed seeds, and a provenance note on every external fact.

## W1.4 as it stood

W1.4 originally called for mutual information estimates with a permutation null across many squad-rounds. The design note replaced that with a single-round diagnostic battery of six statistics combined by calibrated min-p, on the argument that the observation unit is the squad rather than the athlete (30 fully observed rows per round, not 25 scalars) and that the null is fully specified, so exact Monte Carlo p-values are available for any statistic. The battery reports empirical size 0.0533 and power 1.000 from a single round against all five structured alternatives it simulates at full strength.

The verification report in `verification-and-v2.md` checks that claim, corrects four things, and proposes a replacement that answers the prediction question as well as the testing one.
