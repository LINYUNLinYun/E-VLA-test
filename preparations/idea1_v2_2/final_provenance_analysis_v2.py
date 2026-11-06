"""Final scientific analysis plus explicitly indexed throughput execution history."""
import argparse
from pathlib import Path

from common import atomic_json, sha
from benchmark_revision import read
import benchmark_revision_analysis as benchmark
import throughput_provenance_audit_v2 as throughput


def finalize(benchmark_revision, view):
    benchmark_revision, view = Path(benchmark_revision).resolve(), Path(view).resolve()
    parent = Path(read(benchmark_revision/'revision_manifest.json')['source_attempt']).resolve()
    # This must be final, not a live partial provenance snapshot. No GPU work.
    index = throughput.audit(parent, partial=False)
    identities = {str(Path(path).resolve()): sha(path) for path in
        (throughput.__file__, __file__, benchmark.__file__)}
    benchmark.create_view(benchmark_revision, view)
    index_path = view/'throughput_provenance_index.json'
    atomic_json(index_path, index)
    view_receipt = read(view/'benchmark_revision_view_receipt.json')
    view_receipt.update(throughput_provenance_index=str(index_path),
        throughput_provenance_index_sha256=sha(index_path),
        throughput_auditor_source_sha256=sha(throughput.__file__),
        final_analysis_wrapper_source_sha256=sha(__file__))
    atomic_json(view/'benchmark_revision_view_receipt.json', view_receipt)
    route_index = read(parent/'benchmark_revision_index.json')
    route_index.update(view_receipt_sha256=sha(view/'benchmark_revision_view_receipt.json'),
        throughput_provenance_index_sha256=sha(index_path),
        final_analysis_wrapper_source_sha256=sha(__file__))
    atomic_json(parent/'benchmark_revision_index.json', route_index)
    import runtime_analysis
    identities[str(Path(runtime_analysis.__file__).resolve())] = sha(runtime_analysis.__file__)
    benchmark.install_analysis_route(runtime_analysis, view, parent)
    runtime_analysis.analyze(view)
    completion, _ = runtime_analysis.audit(view, strict=True)
    if sha(parent/'episode_ledger.json') != index['ledger_snapshot_sha256']:
        raise ValueError('Ledger changed during final analysis; do not seal this view')
    throughput.verify_index_snapshot(index)
    for path, expected in identities.items():
        if sha(path) != expected:
            raise ValueError('Final analysis code identity changed before sealing')
    # Keep the original analysis_input_receipt bytes: generated tables already
    # bind them. This additional final receipt joins the two verified chains.
    receipt = dict(schema=1, kind='final_scientific_and_execution_provenance_audit',
        complete=completion['complete'] and not index['partial'],
        source_attempt=str(parent), analysis_view=str(view),
        throughput_index_sha256=sha(index_path),
        throughput_auditor_source_sha256=sha(throughput.__file__),
        wrapper_source_sha256=sha(__file__),
        scientific_analysis_source_sha256=sha(runtime_analysis.__file__),
        benchmark_analysis_overlay_source_sha256=sha(benchmark.__file__),
        view_receipt_sha256=sha(view/'benchmark_revision_view_receipt.json'),
        analysis_input_receipt_sha256=sha(view/'analysis_input_receipt.json'),
        analysis_sources_sha256=sha(view/'analysis_sources.json'),
        scientific_completion_audit_sha256=sha(view/'completion_audit.json'),
        parent_ledger_sha256=index['ledger_snapshot_sha256'],
        independent_final_C_acceptance_required=True,
        historical_acceptance_policy='resolve each actual ledger/session SHA to preserved original bytes; only declared exact evidence-path corrections are equivalent')
    atomic_json(view/'final_analysis_provenance_receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-revision', required=True)
    parser.add_argument('--view', required=True)
    args = parser.parse_args()
    finalize(args.benchmark_revision, args.view)


if __name__ == '__main__':
    main()

