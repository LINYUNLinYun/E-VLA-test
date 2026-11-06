"""Run calibration/evaluation, or freeze q using calibration results only."""
from pathlib import Path

from .common import (check_resume, cli_parser, load_config, make_backends, prepare_run, read_jsonl,
                     select_rows, write_json)
from .workflow import freeze_config, integrity_check, run_probe


def main():
    parser = cli_parser(__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and print work count without loading a model")
    parser.add_argument("--freeze-config", type=Path, help="Write a frozen evaluation config; runs no model")
    parser.add_argument("--calibration-results", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.freeze_config:
        if args.calibration_results is None:
            parser.error("--freeze-config requires --calibration-results")
        frozen = freeze_config(config, read_jsonl(args.calibration_results), args.freeze_config,
                               input_path=args.calibration_results)
        print(f"Frozen evaluation config: {args.freeze_config}")
        print(frozen["calibration_provenance"]["selection"])
        return
    rows = select_rows(config, args.limit_groups)
    if args.dry_run:
        from .data import choose_donors
        choose_donors(rows, config["seed"])
        counts = {p["id"]: sum(r["task_type"] == p["task_type"] for r in rows) for p in config["probes"]}
        print({"stage": config["stage"], "samples": len(rows), "samples_per_probe": counts,
               "source_layers": config["source_layers"], "target_layers": {p["id"]: p["target_layers"] for p in config["probes"]},
               "controls": [c["id"] for c in config["controls"]], "conditions": config["conditions"]})
        return
    source, target = make_backends(config)
    out, run = prepare_run(config, rows, source, target, command="probe")
    if (out / "probe_results.jsonl").exists() and not args.resume:
        raise FileExistsError("Results already exist; use --resume or a new output_dir")
    check_resume(out / "run_identity.json", run)
    write_json(out / "model_audit.json", {"source": source.audit(), "target": target.audit(), "identity": run})
    report = integrity_check(source, target, rows[0], config)
    report["identity"] = run
    write_json(out / "patch_integrity.json", report)
    if not report["passed"]:
        raise SystemExit("Integrity failed; results were saved, semantic scan stopped.")
    print(run_probe(config, rows, source, target, out, run, resume=args.resume))


if __name__ == "__main__":
    main()
