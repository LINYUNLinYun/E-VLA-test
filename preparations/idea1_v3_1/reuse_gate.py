"""Quality reuse is opt-in only after same-model paired E2 adoption evidence."""
from .analyze import summarize


def quality_reuse(manifest, requested=None, variant=None, output=None, model=None):
    if variant is not None:
        return bool(variant["environment_reuse"])
    enabled = requested == "on" if requested is not None else bool(manifest.get("quality_environment_reuse", False))
    if enabled:
        report = summarize(manifest, output)
        evidence = [r for r in report["evaluation_optimizations"]
                    if r["condition"] == model+"_full" and r["variant"] == "reuse_env"
                    and r["paired_valid_episodes"] == 8 and r["eligible_for_adoption"]]
        if not evidence:
            raise ValueError("Quality environment reuse requires matching eight-scene E2 reuse_env parity and measured throughput adoption evidence")
    return enabled
