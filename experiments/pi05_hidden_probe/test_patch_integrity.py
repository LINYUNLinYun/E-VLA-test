"""Check native prefix equivalence, self-patching, and layer-0 image replacement."""
__test__ = False

from .common import cli_parser, load_config, make_backends, prepare_run, select_rows, write_json
from .workflow import integrity_check


def main():
    args = cli_parser(__doc__).parse_args()
    config = load_config(args.config)
    rows = select_rows(config, args.limit_groups)
    source, target = make_backends(config)
    out, run = prepare_run(config, rows, source, target, command="integrity")
    report = integrity_check(source, target, rows[0], config)
    report["identity"] = run
    write_json(out / "patch_integrity.json", report)
    print(out / "patch_integrity.json")
    if not report["passed"]:
        raise SystemExit("Integrity failed; inspect differences before semantic interpretation.")


if __name__ == "__main__":
    main()
