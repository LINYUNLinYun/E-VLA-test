"""Render concise report tables from qualified continuous rows, with no guessed data."""
import csv
import json
from pathlib import Path

root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root/"comparison.csv").open()))
summary=json.loads((root/"summary.json").read_text())


def select(run,mode):
    found=[r for r in rows if r["run"]==run and r["phase"]=="continuous" and r["mode"]==mode]
    if len(found)!=1 or found[0]["eligible"]!="True":
        return None
    return found[0]


def n(row,key,digits=2):
    value=row.get(key)
    return "待完成" if value in (None,"") else f"{float(value):.{digits}f}"


def table(headers, values):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"|".join("---" for _ in headers)+"|",
                       *("| "+" | ".join(map(str,row))+" |" for row in values)])


selected=[]
labels=(("main","eager_serial","eager 串行"),("main","eager_dual_prefill_first","eager 双流 Prefill-first"),
        ("main","graph_serial","Graph 串行"),("main","graph_ae_high","Graph 全重叠 AE-first/高优先级"),
        ("vision_wait","graph_ae_high","Graph Vision-only + 上轮等待"),
        ("vision_modulation_wait","graph_ae_high","上述方案 + 条件投影准备"),
        ("vision_modulation_wait_repeat","graph_ae_high","组合方案独立复测"))
for steps in (1,5,10):
    for suffix,mode,label in labels:
        row=select(f"d{steps}_{suffix}",mode)
        selected.append((steps,label,row))

text=["## 主卡端到端延迟", "",
      "所有行来自 GPU2、同一 π0.5 checkpoint、相同观测和噪声规则，每行 100 次连续请求；不同 D 不互为加速比分母。数值、输入与稳态捕获门通过后才显示结果。时效性门以同卡同 D 的 Graph 串行为参考，同时检查观测年龄、Host receipt 年龄、提交/发布/消费速率；eager 因更新率低而不通过不表示其数值错误。", ""]
text.append(table(["D","配置","CPU mean/P50/P95 ms","AE 内部输出 P50 ms","KV 年龄 mean/P95","时效性门"],
    [(steps,label,"待完成","—","—","—") if r is None else
     (steps,label," / ".join(n(r,"cpu_cpu_ms_"+k) for k in ("mean","p50","p95")),n(r,"ae_output_ms_p50"),
      n(r,"observation_age_mean")+" / "+n(r,"observation_age_p95"),r.get("freshness_gate","")) for steps,label,r in selected]))
text += ["", "## 单模块对照", "", "每个模块、每种执行模式各 100 次。CPU→CPU 包括本请求输入准备；单模块 Event 区间才对应该模块的提交/执行跨度。", ""]
isolated=[]
for steps in (1,5,10):
    for module in ("ae","prefill"):
        for mode in ("eager_serial","graph_serial"):
            found=[r for r in rows if r["run"]==f"d{steps}_isolated" and r["phase"]=="isolated_"+module and r["mode"]==mode and r["eligible"]=="True"]
            r=found[0] if len(found)==1 else None
            isolated.append((steps,module,mode,"待完成","—","—") if r is None else
                (steps,module,mode,n(r,"cpu_cpu_ms_p50"),n(r,module+"_cpu_submit_ms_p50"),n(r,module+"_gpu_event_ms_p50")))
text.append(table(["D","模块","模式","CPU→CPU P50 ms","CPU 提交 P50 ms","GPU Event P50 ms"],isolated))
text += ["", "## GPU Event、更新与内存", "",
         "Event 时长包含提交间隙和排队，实际 kernel busy/重叠以 nsys 为准。仅在每请求都提交 Prefill 时，双分支完成时间才表示从本请求计时起点到本请求 AE、Prefill 都结束；缺 Prefill 时原始 both 字段混有仅 AE 完成，表内不将其当公平双分支指标。不能用较早的动作返回时间代替。发布/消费 Hz 只统计同 epoch 相邻响应间的计数变化，排除首响应和 reset bootstrap，分母仍包含 reset 时间。", ""]
text.append(table(["D","配置","AE / Prefill Event P50 ms","Prefill 提交/请求数","双分支完成 P50 ms","发布 / 消费 Hz","峰值 allocated / reserved GiB"],
    [(steps,label,"待完成","—","—","—","—") if r is None else
     (steps,label,n(r,"ae_gpu_event_ms_p50")+" / "+n(r,"prefill_gpu_event_ms_p50"),r["prefill_submissions"]+" / "+r["n"],
      n(r,"both_completion_event_ms_p50") if int(r["prefill_submissions"])==int(r["n"]) else "缺 Prefill，不作同工作量比较",
      n(r,"published_snapshot_updates_hz")+" / "+n(r,"consumed_snapshot_transitions_hz"),
      f"{int(r['peak_allocated_bytes'])/2**30:.2f} / {int(r['peak_reserved_bytes'])/2**30:.2f}") for steps,label,r in selected]))

text += ["", "## 相同准备配置下的串行对照", "",
         "这些 AB/BA 配对两边都开启同一准备优化，区别只在调度；直接串行/并发配对的 ratio 中位数与两组 P50 相除是不同统计量，分别报告。", ""]
paired=[]
for steps in (1,5,10):
    for suffix in ("vision_modulation_wait","vision_modulation_wait_repeat"):
        name=f"d{steps}_{suffix}"
        p=next((p for p in summary["paired"] if p["run"]==name and p["eligible"]),None)
        a,b=select(name,"graph_serial_high"),select(name,"graph_ae_high")
        paired.append((steps,"复测" if suffix.endswith("repeat") else "首次", "待完成","—","—","—") if not(p and a and b) else
            (steps,"复测" if suffix.endswith("repeat") else "首次",f"{p['cpu_speedup']['p50']:.3f}×",
             n(a,"cpu_cpu_ms_p50")+" → "+n(b,"cpu_cpu_ms_p50"),
             n(a,"both_completion_event_ms_p50")+" → "+n(b,"both_completion_event_ms_p50"),
             f"{p['both_event_speedup']['p50']:.3f}×"))
text.append(table(["D","轮次","AB/BA CPU ratio P50","连续 CPU P50 串行→并发 ms","双分支 P50 串行→并发 ms","AB/BA 双分支 ratio P50"],paired))

text += ["", "## 原生 π0.5 JAX baseline", "",
         "同 GPU2、观测序列和显式噪声，每个 D 预热 10 次后测 100 次。JAX 与 PyTorch fresh-serial 都使用本帧新 KV；异步候选使用上一帧完整 KV，语义不同。这里只比较延迟，不声称跨框架逐位相等或闭环成功率等价。原生基线将显式 CUDA noise 生成移到计时外，包含 noise H2D；PyTorch 计时仍含 noise 生成。", ""]
native=[]
for steps in (1,5,10):
    a=next((r for r in summary["native_fresh"] if r["run"]==f"native_d{steps}_gpu2" and r["status"]=="complete"),None)
    fresh=select(f"d{steps}_main","graph_fresh_serial")
    best=next((r["lowest_response_preserving_freshness"] for r in summary["best_candidates"]
               if r["steps"]==steps and r["gpu_uuid"].removeprefix("GPU-")=="83fdf473-8633-29d3-4a42-e482cfdf0c25"),None)
    native.append((steps,"待完成",n(fresh,"cpu_cpu_ms_p50") if fresh else "—","—","—") if not(a and fresh and best) else
       (steps," / ".join(f"{a['cpu_cpu_ms_'+k]:.2f}" for k in ("mean","p50","p95")),n(fresh,"cpu_cpu_ms_p50"),
        f"{best['cpu_p50_ms']:.2f} ({best['run']})",f"{a['cpu_cpu_ms_p50']/best['cpu_p50_ms']:.3f}×"))
text.append(table(["D","JAX CPU mean/P50/P95 ms","PyTorch Graph fresh P50 ms","合格旧 KV 候选 P50 ms","JAX P50 / 候选 P50"],native))
text += ["", "数值门、样本数、Graph 重捕获、各模块提交/等待、完整优化组合与负结果详见 comparison.csv、paired_comparison.csv 和 summary.json。", ""]
(root/"results_tables.md").write_text("\n".join(text))
print("Wrote",root/"results_tables.md")
