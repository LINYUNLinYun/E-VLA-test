"""CPU-only standalone figures, video index and Chinese E2 experiment report."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from preparations.idea1_v3_1.common import ROOT,SUITES,atomic_json

COLORS = dict(full="#2563a6",reuse03="#e68623",d2="#3a9b60")
LABELS = dict(full="Native Full D10",reuse03="Residual 0/3, D10",d2="Native D2")


def read(path):
    return json.loads(Path(path).read_text())


def telemetry(path):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines()]
    return np.asarray([r["step"] for r in rows]),np.asarray([r["eef_pos"] for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    summary = read(args.summary)
    destination = Path(args.output)
    destination.mkdir(parents=True,exist_ok=True)
    figures = destination/"e2_figures"
    figures.mkdir(exist_ok=True)
    outputs = []
    def save(figure,name):
        for extension in ("png","svg"):
            path = figures/f"{name}.{extension}"
            figure.savefig(path,dpi=160,bbox_inches="tight")
            outputs.append(path)
        plt.close(figure)
    modes = ("full","reuse03","d2")
    categories = ("Overall","Spatial","Object","Goal","LIBERO-10")
    figure,axis = plt.subplots(figsize=(11,4.5))
    positions = np.arange(5)
    width = .24
    for j,mode in enumerate(modes):
        q = summary["quality"][mode]
        fractions = [(q["successes"],40)]+[(q["suites"][s]["successes"],10) for s in SUITES]
        bars = axis.bar(positions+(j-1)*width,[100*a/b for a,b in fractions],width,
                        color=COLORS[mode],label=LABELS[mode],alpha=.9)
        axis.bar_label(bars,labels=[f"{a}/{b}" for a,b in fractions],padding=3,fontsize=9)
    axis.set(xticks=positions,xticklabels=categories,ylabel="Success rate (%)",ylim=(0,114),
             title="LIBERO init0: same 40 official task/init pairs")
    handles,labels = axis.get_legend_handles_labels()
    figure.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.045),ncol=3,frameon=False)
    axis.grid(axis="y",alpha=.2)
    axis.set_axisbelow(True)
    figure.subplots_adjust(bottom=.26)
    figure.text(.5,.005,"One init per task; screening counts, not a statistical equivalence test. Full is a verified compatible existing control.",ha="center",fontsize=9)
    save(figure,"quality")
    figure,axes = plt.subplots(1,2,figsize=(11,4))
    timing = summary["timing_summary"]
    for axis,stage in zip(axes,("ae_ms","request_ms")):
        full_mode = "refresh_all" if stage=="ae_ms" else "native"
        values = [timing[full_mode][stage]["p50"]]+[timing[m][stage]["p50"] for m in modes[1:]]
        bars = axis.bar(range(3),values,color=[COLORS[m] for m in modes],alpha=.9)
        axis.bar_label(bars,labels=[f"{x:.2f}" for x in values],padding=3)
        axis.set(xticks=range(3),xticklabels=("Full D10","Residual 0/3","Native D2"),ylabel="p50 latency (ms)",
                 title="AE only: prefilled device KV" if stage=="ae_ms" else "Completed local policy request",ylim=(0,max(values)*1.17))
        axis.grid(axis="y",alpha=.2)
        axis.set_axisbelow(True)
    figure.suptitle("Same physical RTX 4090 / GPU2; 24 real inputs, 3 warmups; compilation excluded")
    figure.text(.5,-.015,"Full AE uses the refresh-all boundary wrapper: raw and decoded outputs were bitwise exact to canonical native D10. Request excludes RPC.",ha="center",fontsize=8.5)
    figure.tight_layout()
    save(figure,"latency")
    videos = []
    for suite in SUITES:
        directories = dict(full=ROOT/"results/idea1_v3_1/run/pi0_full/episodes"/suite/"task00/init00",
            reuse03=ROOT/"results/idea1_v3_2/e2_quality/even/reuse03"/suite/"task00/init00",
            d2=ROOT/"results/idea1_v3_2/e2_quality_d2/even/d2"/suite/"task00/init00")
        figure,axes = plt.subplots(1,3,figsize=(12,3.4))
        for mode in ("full","reuse03"):
            steps,xyz = telemetry(directories[mode]/"telemetry.jsonl")
            for j,axis in enumerate(axes):
                axis.plot(steps,xyz[:,j],color=COLORS[mode],label=LABELS[mode],
                    linestyle="-" if mode=="full" else "--",marker=None if mode=="full" else "o",
                    markevery=max(1,len(steps)//16),markersize=3,alpha=.82,linewidth=1.5)
                axis.set(xlabel="Executed global episode step (20 Hz)",ylabel=f"Measured EEF {'XYZ'[j]} (m)")
                axis.grid(alpha=.2)
        axes[0].legend(fontsize=8,frameon=False)
        figure.suptitle(f"{suite}: fixed task0 / official init0 — actual EEF position")
        figure.tight_layout()
        save(figure,"eef_"+suite)
        row = dict(suite=suite,task=0,init=0,videos={})
        for mode,directory in directories.items():
            video = directory/"video.mp4"
            complete = read(directory/"complete.json")
            if not video.exists() or not complete.get("video_decoded"):
                raise ValueError("Missing decoded fixed video: "+str(video))
            row["videos"][mode] = dict(path=os.path.relpath(video,destination),
                success=read(directory/"result.json")["success"],decoded=True,
                sha256=hashlib.sha256(video.read_bytes()).hexdigest())
        videos.append(row)
    video_index = destination/"e2_video_index.json"
    atomic_json(video_index,dict(fixed_selection="task0/init0 in each of four suites, chosen before outcomes",rows=videos))
    outputs.append(video_index)
    q = summary["quality"]
    reuse_stage = summary["stages"]["reuse03"]
    rows = "".join("<tr><td>"+LABELS[m]+"</td><td>"+str(q[m]["successes"])+"/40</td>"+
        "".join(f'<td>{q[m]["suites"][s]["successes"]}/10</td>' for s in SUITES)+"</tr>" for m in modes)
    video_rows = "".join("<tr><td>"+r["suite"]+" task0/init0</td>"+"".join(
        '<td><a href="'+html.escape(r["videos"][m]["path"])+ '">'+LABELS[m]+(" 成功" if r["videos"][m]["success"] else " 失败")+"</a></td>" for m in modes)+"</tr>" for r in videos)
    figures_html = "".join('<h3>'+suite+' task0/init0</h3><img src="e2_figures/eef_'+suite+'.png">' for suite in SUITES)
    stage_text = json.dumps(summary["stages"],ensure_ascii=False,indent=2)
    paired_text = json.dumps(summary["paired_outcomes"],ensure_ascii=False,indent=2)
    document = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>idea1 v3.2 E2 残差复用实验</title>
<style>body{{max-width:1120px;margin:32px auto;padding:0 20px;font:16px/1.7 system-ui,sans-serif;color:#202936}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd3dc;padding:8px;text-align:left}}pre{{background:#f3f5f8;padding:16px;white-space:pre-wrap;font-size:12px}}a{{color:#2563a6}}</style>
<h1>idea1 v3.2 E2：10 次 Euler 更新，Transformer 仅在 0、3 刷新</h1>
<p>本次 40 个 init0 任务筛查中，Full 与残差复用均为 34/40（85%），额外运行的 canonical D2 为 35/40（87.5%）。残差复用确实降低 Full D10 的计算成本，但在这个小样本中没有显示优于更简单 D2 的收益，不能据此声称统计等价或泛化优势。</p>
<h2>严格数值控制与近似边界</h2><p>同一加载模型、相同真实输入和显式噪声，24 个输入覆盖四个 suite。关闭复用、每步刷新时，raw 和 decoded 输出对 canonical native D10 全部逐位相同；canonical Policy.infer 对照以及 native D2 对照均通过。实际计数为 10 次 Euler 更新、2 次 Transformer 刷新，时间从 1 到约 0（FP32 尾差 −7.45e−8）；穿插其他请求后重复请求的输出差为 0。</p>
<p>缓存精确定义为：18 层 AE Transformer 扫描结束、最终 RMSNorm 之前的输出，减去当步已转 BF16 的 suffix 输入，差值以 FP32 保存。只在更新索引 0、3 运行完整 Transformer。每一步重新计算当前 noisy action、state、time 的 suffix embedding，随后使用当前 embedding 加缓存残差，再重新运行最终 RMSNorm 和 action head；不缓存 velocity。每次请求都重新计算完整图像/VLM prefix，缓存只存在于当次请求内。此次实际验证对象为 pi0。</p>
<h2>成功率与时延</h2><table><tr><th>条件</th><th>总体</th><th>Spatial</th><th>Object</th><th>Goal</th><th>LIBERO-10</th></tr>{rows}</table>
<img src="e2_figures/quality.png"><img src="e2_figures/latency.png">
<p>同一物理 GPU2（RTX 4090）交错测量，24 个真实输入、每条件 3 次 warmup，排除编译。p50 完成本地请求：Full 99.09ms、残差 83.12ms、D2 82.37ms；残差相对 Full 下降 16.1%（1.192×）。AE 单独阶段：严格数值匹配的 Full wrapper 21.99ms、残差 5.63ms、D2 5.04ms，残差下降 74.4%（3.905×）。请求边界包含预处理、全新 prefix、GPU 完成及 decode，排除 RPC；AE 边界使用已准备的设备 KV。历史 Full 成功率控制的运行时间没有被用于端到端加速结论。</p>
<p>Full 是已有兼容的 40 任务控制。新残差 40 次与额外 D2 40 次均核对了官方 init、BDDL、controller、20Hz、H50/R5、checkpoint 身份、native 源码及首请求噪声/seed；80 个新 episode 的首观测 hash 均与对应 Full 完全相同。D2 是本次追加控制，不是历史数据。四个 suite 各 10 个 task，均只使用官方 init0。</p>
<h2>配对结果与阶段描述</h2><pre>{html.escape(paired_text)}</pre><p>下列 final/furthest 阶段是基于接触、抓取及位置等证据的启发式标签；同时列出“Full 成功而候选失败”子集，不能解释为因果失败诊断。</p><pre>{html.escape(stage_text)}</pre>
<h2>真实 EEF XYZ 与执行命令</h2><p>残差相对 Full 的真实 EEF 位置差：每个 episode 刚开始执行的全局步骤 1–5，平均 episode L2 为 {reuse_stage["measured_eef_post_actions1to5_mean_episode_l2_mm"]:.3f}mm。它只覆盖 episode 开头的五步，不是每次 replan 的未来 1–5 offset，也不能说明后期接触/精细阶段的安全性。包含初始状态的 0–5 六个样本为 1.762mm，已分开保存。全程按相同步数对齐的平均 episode L2 为 47.822mm，包含路线、节奏和闭环累积的差异，不是相同观测下的动作误差，也不是 Full 驱动的 shadow 对照。控制器命令保留原始 7 维单位，单独统计，未当作 XYZ 位移。</p>
{figures_html}<h2>预先固定的四场景视频</h2><p>固定选择各 suite 的 task0/init0；Full 蓝线、残差橙色虚线和标记。八个新视频均通过完整 ffmpeg decode 检查，索引另含四个已有 Full 视频。</p><table><tr><th>任务</th><th>Full</th><th>Residual</th><th>D2</th></tr>{video_rows}</table>
<h2>可复查数据</h2><p><a href="e2_summary.json">汇总 JSON</a> · <a href="e2_probe_gpu2.json">实际模型 gate / 时延</a> · <a href="e2_quality_analysis.json">残差 40 对</a> · <a href="e2_d2_quality_analysis.json">D2 40 对</a> · <a href="e2_video_index.json">视频 hash 索引</a></p>
<p>时间预算以约 04:10 UTC 准备开始计，截止 05:40 UTC；GPU 实验于 04:42:04 已清理并释放，随后仅做 CPU 图表与报告。最初一次因 GPU host/container PID 身份未辨明而中断，独立保留为 identity uncertain，不作为质量失败。源码、输入、噪声、checkpoint 身份及结果文件 hash 均有收据；未改 native 模型源码。</p></html>'''
    report = destination/"e2_report_zh.html"
    report.write_text(document)
    outputs.append(report)
    atomic_json(destination/"e2_artifacts.json",dict(
        source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        inputs={str(Path(args.summary)):hashlib.sha256(Path(args.summary).read_bytes()).hexdigest()},
        artifacts={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
        note="CPU-only Matplotlib figures; no camera screenshots; 8 new and 4 existing fully-decoded fixed videos indexed"))
    print(json.dumps(dict(report=str(report),figures=str(figures),video_index=str(video_index)),indent=2))


if __name__ == "__main__":
    main()
