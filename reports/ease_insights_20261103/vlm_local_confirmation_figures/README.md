# EASE local-search heldout confirmation figures

`plot.py` exports the frozen 3-condition × 4-suite success matrix and the two prespecified paired directions. It reads only the three final reports at the fixed paths in the script. Each report must be pinned on the command line by its lowercase SHA256; no partial report discovery or case-level reads are used. Before drawing, the script requires the independent acceptance to pass all 600 cases with zero issues, the supervisor analysis to be complete for 600 cases, and the independent statistics audit to pass with zero differences and to link to the exact acceptance and analysis SHA values.

Run from the repository root after all three reports are final:

```bash
MPLCONFIGDIR=/tmp/ease_local_confirmation_mpl python reports/ease_insights_20261103/vlm_local_confirmation_figures/plot.py \
  --acceptance-sha256 <FINAL_ACCEPTANCE_SHA256> \
  --analysis-sha256 <FINAL_ANALYSIS_SHA256> \
  --statistics-sha256 <FINAL_STATISTICS_SHA256>
```

The command writes into this directory's `exports/`: each of two figures in PNG, PDF, and SVG; one CSV for the 12 matrix cells; one CSV for the 2 directions; and `sha256.json` with source and output digests. SVG keeps text as text and PDF embeds TrueType fonts. `sha256.json` lists every exported figure and CSV; as with any file containing its own digest, it does not include its own SHA256. The command prints that digest on completion.

The matrix reports successes out of 50 for each condition and suite, using the display labels Spatial, Object, Goal, Long and Full, Jac, local_long. The direction figure reports the existing success difference, task-cluster 95% CI in percentage points, exact task-level sign-flip Holm-2 p value, 50 paired cases per direction, and whether the frozen reproduction rule passed. Its x-axis includes zero, both estimates, and both CI endpoints with at least 3 percentage points or 15% span padding. Both figures and `sha256.json` state the overall analysis conclusion; the gate requires the supervisor and independent conclusions to agree. A direction can pass its own reproduction rule while the overall conclusion is `not_reproduced` because overall candidate eligibility also contributes to that verdict. Heldout initial states are 39–43 across 10 task clusters. The plots make no claim about quality noninferiority, exclusive-GPU acceleration, or runtime-adaptive benefit. The script does not recompute intervals or tests.
