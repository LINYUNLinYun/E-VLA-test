#!/usr/bin/env python3
"""按论文 Table 2 的口径汇总 FLASH 复现的 FR / Acc / 延迟。

口径（与论文 Table 2 caption 对齐）：
  FR  = flash-path 轮数 / 全部 replan 轮数
  Acc = flash-path 轮的接受前缀长度均值 / replan 窗口（12）
  Lat = 每轮的推理延迟（服务端 total_ms 口径，与论文的 flash-path round 成本同边界）
  /Act= 每轮的推理延迟 / 该轮实际执行的步数（draft 轮取接受前缀长度，全路径轮取 12），再对轮次求均值

每个 task 的第 0 集第一轮含一笔 5–15 s 的一次性成本（成因：Triton 运行时池按 prompt 的 token 长度缓存
runtime，长度变化时 evict + 重建 + 重做 CUDA graph capture），默认剔除。

用法:
  python3 scripts/flash_fr_acc.py 'policies/realtime-vla-flash/data/test/paper_libero_*'
  python3 scripts/flash_fr_acc.py <dir> --keep-first-episode
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics
import sys
from collections import defaultdict

REPLAN = 12


def _load_rounds(run_dir: pathlib.Path, keep_first: bool) -> list[dict]:
    rows = []
    for ep_dir in sorted(run_dir.glob("episodes/*/infer.jsonl")):
        recs = []
        for line in ep_dir.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                recs.append(json.loads(line))
        if not recs:
            continue
        if not keep_first and recs[0].get("episode_idx") == 0:
            continue
        rows.extend(recs)
    return rows


def _stats(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return (float("nan"), float("nan"))
    return (statistics.fmean(vals), statistics.median(vals))


def summarize(run_dir: pathlib.Path, keep_first: bool) -> dict | None:
    rounds = _load_rounds(run_dir, keep_first)
    if not rounds:
        return None
    suite = rounds[0].get("task_suite_name", run_dir.name)
    draft = [r for r in rounds if r.get("route_type") == "draft"]
    full = [r for r in rounds if r.get("route_type") == "full"]

    acc = [r.get("accepted_prefix_len", 0) / REPLAN for r in draft]
    lat_all = [r["total_ms"] for r in rounds if r.get("total_ms") is not None]
    lat_draft = [r["total_ms"] for r in draft if r.get("total_ms") is not None]
    lat_full = [r["total_ms"] for r in full if r.get("total_ms") is not None]
    rt = [r["client_roundtrip_ms"] for r in rounds if r.get("client_roundtrip_ms") is not None]

    per_act = []
    for r in rounds:
        lat = r.get("total_ms")
        if lat is None:
            continue
        steps = r.get("accepted_prefix_len") if r.get("route_type") == "draft" else REPLAN
        steps = int(steps or 0)
        if steps > 0:
            per_act.append(lat / steps)

    return {
        "suite": suite,
        "run": run_dir.name,
        "rounds": len(rounds),
        "fr": len(draft) / len(rounds),
        "acc": statistics.fmean(acc) if acc else float("nan"),
        "lat_all": _stats(lat_all),
        "lat_draft": _stats(lat_draft),
        "lat_full": _stats(lat_full),
        "roundtrip": _stats(rt),
        "per_act": _stats(per_act),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--keep-first-episode", action="store_true")
    args = ap.parse_args()

    dirs = [pathlib.Path(p) for p in sorted(glob.glob(args.pattern))]
    dirs = [d for d in dirs if d.is_dir()]

    print("| suite | run | 轮数 | FR | Acc | Lat. mean/median (ms) | draft 轮 mean | full 轮 mean | /Act | 客户端往返 mean/median |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    agg = []
    for d in dirs:
        s = summarize(d, args.keep_first_episode)
        if s is None:
            continue
        agg.append(s)
        print(
            f"| {s['suite']} | {s['run']} | {s['rounds']} | {s['fr']*100:.1f}% | {s['acc']*100:.1f}% | "
            f"{s['lat_all'][0]:.1f} / {s['lat_all'][1]:.1f} | {s['lat_draft'][0]:.1f} | {s['lat_full'][0]:.1f} | "
            f"{s['per_act'][0]:.2f} | {s['roundtrip'][0]:.1f} / {s['roundtrip'][1]:.1f} |"
        )

    by_suite = defaultdict(list)
    for s in agg:
        by_suite[s["suite"]].append(s)
    if len(by_suite) > 1 or len(agg) > 1:
        print()
        print("| suite | FR（轮加权） | Acc（轮加权） | Lat. mean (ms) | /Act |")
        print("| --- | --- | --- | --- | --- |")
        for suite, ss in sorted(by_suite.items()):
            rounds = sum(x["rounds"] for x in ss)
            fr = sum(x["fr"] * x["rounds"] for x in ss) / rounds
            acc = sum(x["acc"] * x["rounds"] for x in ss) / rounds
            lat = sum(x["lat_all"][0] * x["rounds"] for x in ss) / rounds
            pa = sum(x["per_act"][0] * x["rounds"] for x in ss) / rounds
            print(f"| {suite} | {fr*100:.1f}% | {acc*100:.1f}% | {lat:.1f} | {pa:.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
