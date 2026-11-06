"""Extract only the configured source positions/layers into identity-bound NPZ files."""
from .common import (FeatureCache, cli_parser, load_config, make_backends, prepare_run,
                     select_rows, source_layers, write_json)
from .data import load_observation


def main():
    args = cli_parser(__doc__).parse_args()
    config = load_config(args.config)
    rows = select_rows(config, args.limit_groups)
    source, target = make_backends(config)
    out, run = prepare_run(config, rows, source, target, command="extract")
    layers = source_layers(config, source)
    cache = FeatureCache(config.get("cache_dir", out / "source_cache"), run["source_identity"])
    index = []
    for row in rows:
        obs = load_observation(row)
        for probe in config["probes"]:
            if row["task_type"] != probe["task_type"]:
                continue
            _, positions, key = cache.get(source, obs, probe["source_selector"], layers, row.get("object_text"))
            index.append({"sample_id": row["sample_id"], "probe_id": probe["id"], "cache_key": key,
                          "layers": layers, "positions": positions})
        print(f"extracted {row['sample_id']}", flush=True)
    write_json(out / "source_index.json", {"identity": run, "features": index})


if __name__ == "__main__":
    main()
