# Draft source generalization figures

Status: **complete_exploratory_discovery**, 360/360 episodes. Discovery remains exploratory; assess the selected differences using independent confirmation.

Rows are draft source; columns are fixed target suite. Every observed cell shows completed episode count. White missing cells are not failures or zero rates. The orange outline identifies the target-matching source.

`flash_rejection_full` counts rejection-triggered Full requests; `flash_all_full` additionally includes initial/periodic refresh. Acceptance is conditional on actual verified draft rounds and is never reported for pure draft.

Independent confirmation is pending unless separate held-out analysis and frozen selection are supplied. Its primary cost plot uses complete native policy-call milliseconds per actually executed action, with client RPC milliseconds per action marked separately. Rejected requests that execute zero actions still incur full cost. Mean request latency alone cannot establish acceleration; GPU stages are not end-to-end. Real episode wall and success counts appear alongside service cost. Cold requests and changed closed-loop trajectories remain included.

Regenerate discovery: `python3 preparations/ease_insights_20261103/draft/plot_results.py --analysis <analyze-output> --output reports/ease_insights_20261103/draft_figures`. Add `--replication-analysis <heldout-analyze-output> --selection <frozen-selection.json>` for the independent comparison. Discovery accepts only init20; replication accepts only init21–24.

Input file SHA256 values and metric definitions: [figure_provenance.json](figure_provenance.json). Exact plotted cells: [matrix_cells.json](matrix_cells.json). PNG and PDF are emitted for every figure.
