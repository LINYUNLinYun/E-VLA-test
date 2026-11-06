"""Inspect the actual loaded checkpoint and first manifest observation."""
from .common import cli_parser, load_config, make_backends, prepare_run, select_rows, write_json
from .data import load_observation


def main():
    args = cli_parser(__doc__).parse_args()
    config = load_config(args.config)
    rows = select_rows(config, args.limit_groups)
    source, target = make_backends(config)
    out, run = prepare_run(config, rows, source, target, command="audit")
    obs = load_observation(rows[0])
    report = {"source": source.audit(source.prepare_source(obs)),
              "target": target.audit(), "identity": run,
              "note": "Audit records implementation/weights; it does not establish text capability."}
    write_json(out / "model_audit.json", report)
    print(out / "model_audit.json")


if __name__ == "__main__":
    main()
