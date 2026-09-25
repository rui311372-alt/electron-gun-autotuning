# Electron Gun Autotuning

**English** | [简体中文](README.zh-CN.md)

Image-feedback autotuning of a real microfocus electron gun, with Bayesian candidate selection and offline identification of a stability-constrained optimal point.

This repository contains experimental control scripts, a hardware-independent offline example, and selected experimental tables and reports. **It does not yet provide the complete raw-image dataset or a ready-to-run hardware installation.**

## Overview

The method has two stages:

1. **Online acquisition and search.** Device control, feedback checks, repeated image acquisition, image evaluation, and Bayesian candidate selection form an automated measurement loop.
2. **Offline optimal-point selection.** Saved measurements are used to identify a low-variability branch connected to the highest sampled cathode voltage, then select the measured candidate with the lowest score within that branch.

The online stage collects evidence and recommends candidates. The offline stage determines the final optimal point; it does not control the gun or rerun the Gaussian process.

### What the score measures

The image-analysis code combines two orthogonal edge full widths at half maximum (FWHMs):

$$
\mathrm{score}=\sqrt{x_1^2+x_2^2}.
$$

The widths and score are expressed in pixels (px). Under the same imaging and processing conditions, a smaller score represents smaller combined image-edge widths. This is a focus-related image metric, not a measurement of every aspect of beam quality or a calibrated physical spot diameter.

For the default offline analysis, each candidate has three image scores. The **candidate score is the mean of the two scores with the smallest relative difference**. The coefficient of variation (**CV**) uses **all three scores** and the sample standard deviation. A low candidate score alone is therefore not sufficient for selection.

## Repository layout

```text
electron-gun-autotuning/
├── README.md                         English project guide
├── README.zh-CN.md                    Chinese project guide
├── code/
│   ├── offline/
│   │   ├── run_offline.py             Example runner
│   │   ├── stable_working_point_round10.py
│   │   ├── candidate_data.csv         Four-run offline example
│   │   ├── reference_results.json     Archived reference results
│   │   ├── run_example.cmd            Windows launcher
│   │   └── 使用说明.md                 Detailed Chinese instructions
│   └── online/                       Device, camera, analysis, and search scripts
├── data/                             Experimental tables and reports
└── docs/PACKAGING_NOTES.md            Packaging scope and release checklist
```

## Quick start: offline example

**Requirements:** Python 3.10 or later. This offline example uses only the Python standard library. It requires no camera, electron gun, vendor SDK, or third-party Python packages.

Open a terminal in the repository root and run:

```bash
python -X utf8 code/offline/run_offline.py
```

On Windows, you can also double-click `code/offline/run_example.cmd` if Python is installed and available on `PATH`.

To analyze a different, prevalidated CSV, provide its path:

```bash
python -X utf8 code/offline/run_offline.py "path/to/your/candidate_data.csv"
```

The runner creates a new `code/offline/results/<timestamp>/` directory for each invocation. It also runs sensitivity and prefix analyses, so it takes longer than optimal-point selection alone. Generated results are ignored by Git; the archived `reference_results.json` is tracked.

| Output | Contents |
| --- | --- |
| `optimal_points.csv` | Optimal voltages, candidate scores, branch-entry intervals, and summary statistics |
| `segments.csv` | Voltage segments and their high-/low-variability labels |
| `results.json` | Main-analysis settings, results, and excluded input rows |
| `sensitivity.csv` | Results under alternative aggregation, segmentation, and other analysis settings |
| `prefix.csv` | Prefix analysis for eligible long-budget runs; may be absent for other inputs |

### Input format

Use a UTF-8 CSV with one candidate per row and the following columns:

| Column | Meaning |
| --- | --- |
| `run` | Run identifier; keep independent runs separate |
| `iteration` | Integer acquisition order within that run |
| `uc_set` | Cathode-voltage setpoint in V; used for sorting and optimal-point reporting |
| `uc_feedback` | Recorded voltage feedback in V; retained separately from the setpoint |
| `shot1`, `shot2`, `shot3` | Scores from three image acquisitions at that candidate, in px |

Additional columns may be retained. The main analysis recomputes candidate score and CV from `shot1`–`shot3`, rather than trusting stored summary columns. It does **not** extract image features, validate the original images, or reapply the online 10% closest-pair consistency criterion. Apply the experiment's validity and consistency checks before supplying new data.

### Reference example

The bundled CSV contains **180 valid candidates from four runs**, not the complete nine-condition dataset or the raw records for all ten repeat runs.

| Run identifier | Valid candidates | Optimal cathode voltage (V) | Candidate score (px) |
| --- | ---: | ---: | ---: |
| Short 1 (`短预算1`) | 30 | 631 | 31.40455 |
| Short 2 (`短预算2`) | 30 | 638 | 32.10840 |
| Long 1 (`长预算1`) | 60 | 622 | 30.87535 |
| Long 2 (`长预算2`) | 60 | 620 | 31.83380 |

The table follows [the archived reference results](code/offline/reference_results.json). It is a check for the supplied offline example, not a benchmark demonstrating superiority over other optimizers.

**Verification:** On 2026-09-25, the unchanged core script was run directly on Windows with Python 3.11.9 and the bundled CSV. All 180 rows were accepted, and all four optimal voltages and scores matched the table. Five output files were produced, including 119 sensitivity rows with `ok` status and eight prefix rows. One clustering-center value differed from the reference by about `2.22e-16` (floating-point roundoff); the JSON files are not byte-identical. The convenience runner was inspected, not separately executed during this check. No hardware was accessed.

## Offline selection method and interpretation

Within each run, the default analysis:

1. Orders candidates by `uc_set` and computes CV as a percentage, then `ln(CV + 1e-6)`.
2. Uses dynamic programming with a Laplace segment cost and BIC to select contiguous segments; the default minimum segment size is four candidates.
3. Clusters segment medians into two variability states, weighted by the number of candidates in each segment.
4. Retains consecutive low-variability segments starting from the highest sampled voltage and moving downward, stopping at a high-variability segment.
5. Selects the measured candidate with the lowest candidate score in this terminal branch. Ties in score favor lower voltage.

“Low variability” is relative to the supplied run. The main analysis has neither a fixed 5% CV cutoff nor a fixed 40% lower search bound. Historical output fields `transition_lower_uc` and `transition_upper_uc` denote the **entry interval of the terminal low-variability branch**, not a unique physical transition or a confidence interval.

Known boundaries of the preserved implementation:

- Too few candidates, insufficient segmentation, or a highest-voltage segment classified as high variability can prevent optimal-point selection. An output does not constitute a statistical test proving distinct physical states.
- The simplified Laplace cost differs from the full likelihood when its scale floor (`1e-6`) is active. The detailed instructions record that this floor was not reached in the relevant intervals of the bundled example. Check this implementation boundary before using new data with repeated or nearly identical CVs.
- The selected point is the best **measured** candidate in the identified branch. It is not a guaranteed mathematical global minimum or a certified long-term operating setpoint.

See [the detailed offline instructions (Chinese)](code/offline/使用说明.md) for implementation notes.

## Experimental data and availability

Original experimental category names are retained. Files in different folders can describe the same measurements; file counts and summary tables must not be added together as independent sample sizes.

| Material | Repository location | Available through Git |
| --- | --- | --- |
| Four-run candidate-level example | `code/offline/candidate_data.csv` | CSV and reference results |
| Detailed fixed-condition runs | `data/固定工况下，最优点细致测试/` | Workbooks, JSON records, text reports, and selected PDF plots; not the raw/fit image files |
| Ten-run summary | `data/10次闭环运行最优点与score.xlsx` and the corresponding ten-run subfolder | Summary workbooks, not all ten runs' raw records |
| High-voltage stability test | `data/高Uc稳定性测试/` | Workbook |
| Power-cycle retest | `data/断电测试/` | Separate workbook |
| Same-image fitting test | Author's local `data/同样图片拟合测试/` | Not included; current files are ignored images |
| Nine-condition matrix test | Author's local `data/3乘3矩阵测试/` | No files supplied in this folder |

The high-voltage stability test and the power-cycle retest are **separate datasets**. Power-cycle summaries use the arithmetic mean of three image scores, unlike the closest-pair mean used for optimal-point selection. The workbook field “三次相对差” (relative range) is not CV.

Raw and fitted-image files are retained in the author's local copy but excluded by [.gitignore](.gitignore). A complete image archive and download link have not been provided. A fresh clone therefore supports the supplied tabular offline example, **not full image-to-result reproduction**. Further provenance details are in [the data guide (Chinese)](data/README.md).

## Online scripts and hardware requirements

`code/online/` preserves the supplied control, camera, image-processing, and optimization scripts. It includes several `bayes_optlr_*.py` variants; the script and configuration actually used for each experiment must be confirmed before hardware reproduction.

**Do not launch these scripts as an offline demo.** They can communicate with real high-voltage equipment. Hardware use requires authorized operators, device-specific configuration, the appropriate camera drivers/SDK, and the laboratory's safety procedures.

The existing [online dependency file](code/online/requirements.txt) is not a complete locked environment for all variants. For example, Bayesian scripts import `skopt`, whose package is `scikit-optimize`, but that package is not listed there. The camera SDK and binary wrapper are excluded from Git pending redistribution permission and compatibility checks. No online hardware workflow has been validated as part of this documentation update.

## License, citation, and release status

No code or data license is assigned in this repository yet. Public redistribution and reuse terms require confirmation by the authors and laboratory; third-party camera components have separate rights. This repository should not be described as a complete open-source release until those matters are resolved.

No paper DOI or finalized citation is supplied here. Add the confirmed paper citation and a versioned code/data release when available. See [packaging notes and release checks (Chinese)](docs/PACKAGING_NOTES.md).
