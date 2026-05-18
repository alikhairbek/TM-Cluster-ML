# TM-Cluster-ML# TM-Cluster-ML  (rebuilt project, v2)

Leakage-controlled machine-learning benchmark of transition-metal
nanocluster stability, built on the Quantum Cluster Database
(Manna et al., Sci. Data 2023). Nine metals, N <= 55.

## Files

| file | purpose |
|---|---|
| `descriptors.py`   | physical structural descriptors (gyration tensor, coordination, bond lengths, Steinhardt q4/q6/q8). Run directly for unit tests. |
| `run_analysis.py`  | full pipeline: targets, leakage-controlled evaluation, feature ablation, leave-one-element-out, change-point structural analysis, magic numbers. |
| `requirements.txt` | pinned dependencies. |
| `Data9M.csv`       | QCD export (place here; not redistributed). |

## How to run

```bash
pip install -r requirements.txt
python descriptors.py        # sanity check (known geometries)
python run_analysis.py       # full study; descriptors are cached after first run
```

Outputs go to `RESULTS_v2/Tables` and `RESULTS_v2/Figures`.

## What each result means (for the manuscript)

* **Target A (total energy / atom).** A two-parameter per-element fit
  `a + b/N` already reaches R^2 ~ 0.99. Report this as *motivation*:
  the commonly used target is uninformative; high headline R^2 on it
  does not demonstrate learning.

* **Target B (relative isomer energy / atom).** The real task. The
  ablation shows geometry-only R^2 ~ 0.66 with the new descriptors
  (vs ~0.28 with the old coarse scalars), rising to ~0.72 with
  composition. Size adds nothing once energy is per atom. This is the
  central, positive contribution.

* **Leave-one-element-out.** Relative stability is *largely*
  transferable across the series (R^2 ~ 0.55-0.67 for most held-out
  metals) but breaks for Co and Pt -> element-specific physics worth
  discussing (Co magnetism; Pt relativistic 5d effects).

* **Compaction onset.** Change-point on the gyration planarity index
  gives a clean, ordered trend: 3d ~ N18, 4d ~ N13, 5d ~ N8. Present
  descriptively as size-dependent structural compaction, not as a
  sharp "2D->3D phase transition".

* **Magic numbers.** Second finite difference of global-minimum
  energies, single stated prominence threshold (0.03 eV/atom).
  State threshold sensitivity in Methods.

## Data / code availability (do this before resubmission)

Deposit `descriptors.py`, `run_analysis.py`, the cached descriptors,
and `RESULTS_v2/` on Zenodo and cite the **DOI** in the paper.
Do not use "link upon acceptance".
