"""Evidence report with separate native denominators and projection stress rows."""
import argparse
from collections import defaultdict
import csv
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import quote
import numpy as np
from .transfer_protocol import MODES, ENVIRONMENTS, CASE_COUNTS, load_manifest, write_json, sha, complete_result

STYLES = {'full_d10': dict(color='tab:blue', linestyle='-', marker='.', alpha=.65),
          'v14_d10': dict(color='tab:orange', linestyle='--', marker='o', alpha=.65),
          'draft_goal': dict(color='tab:green', linestyle='-.', alpha=.65),
          'draft_object': dict(color='tab:red', linestyle=':', alpha=.65),
          'draft_spatial': dict(color='tab:purple', linestyle='-', alpha=.65),
          'draft_10': dict(color='tab:brown', linestyle='--', alpha=.65)}


def read_jsonl(path):
    if not path or not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def resolve_path(root, attempt, value):
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    if (attempt/p).exists():
        return attempt/p
    return root/p


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def full_decode(path):
    """Read the complete stream with ffmpeg, never infer validity from metadata."""
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                            '-show_entries', 'stream=width,height,nb_read_frames,avg_frame_rate,duration',
                            '-of', 'json', str(path)], capture_output=True, text=True)
    decode = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:v:0', '-f', 'null', '-'],
                             capture_output=True, text=True)
    streams = json.loads(probe.stdout or '{}').get('streams', []) if probe.returncode == 0 else []
    return dict(video=str(path), sha256=sha(path), full_stream_decode=decode.returncode == 0,
                decode_error=decode.stderr or None, probe_status=probe.returncode, streams=streams)


def summarize(manifest, records):
    summaries, paired = [], []
    for env in ENVIRONMENTS:
        support = manifest['environment_receipt']['environments'][env]
        for mode in MODES:
            rows = [r for r in records if r['environment'] == env and r['mode'] == mode]
            complete = [r for r in rows if r['status'] == 'complete' and isinstance(r.get('success'), bool)]
            successes = sum(r['success'] for r in complete)
            summaries.append(dict(environment=env, mode=mode, planned=CASE_COUNTS[env], complete=len(complete),
                                   successes=successes, success_rate=successes/len(complete) if complete else None,
                                   technical=sum(r['status'] == 'technical_failure' for r in rows),
                                   pending=CASE_COUNTS[env]-sum(r['status'] in ('complete', 'technical_failure', 'unsupported') for r in rows),
                                   unsupported=sum(r['status'] == 'unsupported' for r in rows), support=support['status'],
                                   result_scope='2D projection stress' if env == 'pusht' else 'native task oracle'))
        baseline = {r['case']: r for r in records if r['environment'] == env and r['mode'] == 'full_d10' and r['status'] == 'complete'}
        for mode in MODES[1:]:
            other = {r['case']: r for r in records if r['environment'] == env and r['mode'] == mode and r['status'] == 'complete'}
            keys = sorted(set(baseline)&set(other))
            paired.append(dict(environment=env, mode=mode, paired_cases=len(keys),
                               full_success_to_condition_failure=sum(baseline[k]['success'] and not other[k]['success'] for k in keys),
                               full_failure_to_condition_success=sum(not baseline[k]['success'] and other[k]['success'] for k in keys),
                               success_delta_pp=100*np.mean([int(other[k]['success'])-int(baseline[k]['success']) for k in keys]) if keys else None,
                               full_zero_inconclusive=not any(r['success'] for r in baseline.values())))
    return summaries, paired


def report(output, decode=True):
    output = Path(output).resolve(); manifest = load_manifest(output/'manifest.json', verify_sources=False)
    directory = output/f'report_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}'
    directory.mkdir(exist_ok=False)
    all_results = defaultdict(list)
    for path in output.glob('episodes/*/case*/*/attempt*/result.json'):
        r = json.loads(path.read_text()); r['_result_path'] = str(path); r['_result_sha256'] = sha(path)
        all_results[(r.get('environment'), r.get('case'), r.get('mode'))].append(r)
    selected = []
    for case in manifest['cases']:
        for mode in MODES:
            key = (case['environment'], case['case'], mode)
            rows = sorted([r for r in all_results.get(key, []) if r.get('manifest_hash') == manifest['manifest_hash']], key=lambda r: r['_result_path'])
            complete = [r for r in rows if complete_result(r, manifest, case, mode)]
            if len(complete) > 1:
                raise ValueError('Duplicate matching complete keys: '+str(key))
            if complete:
                selected.append(complete[0])
            elif rows:
                selected.append(rows[-1])
            elif case['support_status'] == 'unsupported':
                selected.append(dict(environment=case['environment'], case=case['case'], mode=mode,
                                     status='unsupported', success=None, reason=manifest['environment_receipt']['environments'][case['environment']].get('reason')))
    summary, paired = summarize(manifest, selected)
    write_csv(directory/'success.csv', summary); write_csv(directory/'paired.csv', paired)
    videos, trajectories, timing, route_rows = [], [], [], []
    for r in selected:
        if '_result_path' not in r:
            continue
        attempt = Path(r['_result_path']).parent
        path = resolve_path(output, attempt, r.get('telemetry_path', r.get('trace_path')))
        trace = read_jsonl(path)
        if trace:
            positions = []
            for row in trace:
                actual = row.get('actual', row)
                value = actual.get('eef_xyz_world', actual.get('agent_xy_pixels', actual.get('eef_pos', actual.get('xy'))))
                if value is not None:
                    positions.append(value)
            trajectories.append(dict(environment=r['environment'], case=r['case'], mode=r['mode'], positions=positions,
                                     status=r['status'], source=str(path), sha256=sha(path),
                                     quantity='actual measured XY' if r['environment'] == 'pusht' else 'actual measured EEF XYZ'))
        if r['case'] == 0:
            path = resolve_path(output, attempt, r.get('video_path'))
            if path and path.is_file():
                try:
                    receipt = full_decode(path) if decode else dict(video=str(path), full_stream_decode=None)
                except Exception as error:
                    receipt = dict(video=str(path), full_stream_decode=False, error=str(error))
                videos.append(dict(environment=r['environment'], case=0, mode=r['mode'],
                                   relative_video=quote(os.path.relpath(path, directory), safe='/'), **receipt))
            else:
                videos.append(dict(environment=r['environment'], case=0, mode=r['mode'], full_stream_decode=False, error='Fixed video missing'))
        request_path = resolve_path(output, attempt, r.get('inference_path', r.get('infer_path')))
        requests = read_jsonl(request_path)
        for q in requests:
            response = q.get('response', q)
            item = dict(environment=r['environment'], case=r['case'], mode=r['mode'], gpu_uuid=r.get('gpu_uuid'),
                        request=q.get('request', q.get('replan')), category=response.get('category', 'quality'),
                        episode_first_request=q.get('request') == 0, startup_events=bool(response.get('startup_events')),
                        client_completed_ms=q.get('client_completed_s', 0)*1000)
            for k, v in response.get('timing', response.get('timings', {})).items():
                if isinstance(v, (float, int)):
                    item[k] = v
            timing.append(item)
            route_rows.append(dict(environment=r['environment'], case=r['case'], mode=r['mode'],
                                   gpu_uuid=r.get('gpu_uuid'), request=q.get('request'),
                                   category=response.get('category'), input_hash=response.get('input_hash'),
                                   noise_hash=response.get('noise_hash'), **response.get('observed_calls', {})))
    save_trajectories = directory/'trajectories.json'
    write_json(save_trajectories, trajectories); write_json(directory/'video_decode.json', videos)
    media_index(directory, videos)
    write_csv(directory/'request_timing.csv', timing)
    write_csv(directory/'observed_routes.csv', route_rows)
    latency = []
    for gpu in sorted({r['gpu_uuid'] for r in timing if r.get('gpu_uuid')}):
        for env in ENVIRONMENTS:
            for mode in MODES:
                rows = [r for r in timing if r['gpu_uuid'] == gpu and r['environment'] == env and r['mode'] == mode
                        and r['category'] == 'quality' and not r['episode_first_request'] and not r['startup_events']]
                if rows:
                    a = [r['client_completed_ms'] for r in rows]
                    latency.append(dict(gpu_uuid=gpu, environment=env, mode=mode, quality_requests=len(rows),
                                        client_p50_ms=float(np.percentile(a, 50)), client_p95_ms=float(np.percentile(a, 95)),
                                        comparison_scope='within same GPU shard only; episode-first and recorded startup excluded'))
    write_csv(directory/'latency_within_shard.csv', latency)
    plots(directory, summary, trajectories)
    text = ['# v3.2 E4 零样本跨环境实验', '',
            '以下条件固定使用原发布权重，无目标环境训练、适配或事后挑选head。分母只包含原生任务正常完成的有效episode；技术失败、未运行与不支持接口单列。PushT是二维投影压力测试，不与机械臂任务合并成功率。', '',
            '| 环境 | 条件 | 成功/有效 | 成功率 | 技术失败 | 待运行 | 支持状态 |', '|---|---|---:|---:|---:|---:|---|']
    for r in summary:
        sr = f'{r["success_rate"]*100:.1f}%' if r['success_rate'] is not None else 'N/A'
        text.append(f'| {r["environment"]} | {r["mode"]} | {r["successes"]}/{r["complete"]} | {sr} | {r["technical"]} | {r["pending"]} | {r["support"]} |')
    text += ['', 'Full也是零样本桥接部署的控制组：若该环境Full有效样例全部失败，结果只能表明本冻结部署未成功，无法将draft失败归因于模型容量或泛化能力。原生控制频率、末端定义、视角和规范化分布差异均是保留的混杂因素。', '',
             'CALVIN只覆盖8条debug语言段起始状态，不是正式长序列评测；Simpler固定pick coke can与5个seed；PushT固定5seed、512px↔0.6m人工平面，未执行Z/旋转/夹爪。源LIBERO为20Hz，CALVIN保留30Hz、Simpler保留Google原生3Hz、PushT保留10Hz；时间语义差异仍是混杂因素。六条件每个case始终同一物理卡，延迟只允许同GPU分片内比较，启动/编译不并入稳态速度。', '',
             '![各环境成功率，分母独立](success.png)', '', '![预先固定case0真实轨迹](trajectories.png)', '',
             '[打开固定case0全部条件的视频索引与播放器](media_index.html)。视频使用相对路径，按预先指定case选择，不按成功或失败筛选。', '',
             '原始清单与证据：manifest.json冻结源码、checkpoint元数据、语言证明、环境资产、坐标映射、native成功oracle和cap；本报告video_decode.json完整解码固定case0视频；trajectories.json保留实际XYZ/XY及来源hash。没有用命令累加冒充实际轨迹。', '',
             f'匹配有效episode {sum(r["complete"] for r in summary)}/108；本快照未将缺项重新定义为策略0分。', '']
    (directory/'report_zh.md').write_text('\n'.join(text))
    write_json(directory/'receipt.json', dict(status='complete', manifest_hash=manifest['manifest_hash'],
               result_files={r['_result_path']: r['_result_sha256'] for r in selected if '_result_path' in r},
               summary=summary, paired=paired, report_scope='No pooled cross-environment capacity score',
               files={p.name: sha(p) for p in directory.iterdir() if p.is_file()}))
    print(json.dumps(dict(report=str(directory), complete=sum(r['complete'] for r in summary)), indent=2))


def media_index(directory, videos):
    rows = ['<!doctype html><meta charset="utf-8"><title>E4 fixed case0 media</title>',
            '<style>body{font:15px sans-serif;margin:2em}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.6em}video{width:320px;max-height:240px}</style>',
            '<h1>E4 固定 case0 视频与真实轨迹</h1><p>每环境全部六条件；缺失与解码失败明确保留。PushT 是二维接口压力测试。</p>',
            '<p><a href="report_zh.md">中文报告</a> · <a href="success.png">独立分母成功率图</a> · <a href="trajectories.png">实际XYZ/XY轨迹图</a></p>',
            '<img src="trajectories.png" style="max-width:100%"><table><tr><th>环境</th><th>条件</th><th>MP4 / 播放器</th><th>完整解码</th></tr>']
    for env in ENVIRONMENTS:
        for mode in MODES:
            r = next((r for r in videos if r['environment'] == env and r['mode'] == mode), None)
            relative = r.get('relative_video') if r else None
            media = f'<a href="{html.escape(relative, quote=True)}">打开MP4</a><br><video controls preload="metadata" src="{html.escape(relative, quote=True)}"></video>' if relative else '未完成或无视频'
            status = '通过' if r and r.get('full_stream_decode') else ('失败' if r else '待运行/不支持')
            rows.append(f'<tr><td>{env}</td><td>{mode}</td><td>{media}</td><td>{status}</td></tr>')
    rows.append('</table>')
    (directory/'media_index.html').write_text('\n'.join(rows))


def plots(directory, summary, trajectories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    for ax, env in zip(axs, ENVIRONMENTS):
        rows = [r for r in summary if r['environment'] == env]
        ax.bar([r['mode'] for r in rows], [r['success_rate'] if r['success_rate'] is not None else 0 for r in rows],
               color=[STYLES[r['mode']]['color'] for r in rows], alpha=.65)
        for i, r in enumerate(rows):
            ax.text(i, .03, f'{r["successes"]}/{r["complete"]}' if r['complete'] else 'N/A', ha='center', fontsize=8)
        ax.set_ylim(0, 1.08); ax.tick_params(axis='x', rotation=60); ax.set_title(env+(' (2D stress)' if env == 'pusht' else ''))
        ax.set_ylabel('Native success fraction'); ax.grid(axis='y', alpha=.2)
    fig.tight_layout(); fig.savefig(directory/'success.png', dpi=160); plt.close(fig)
    fig = plt.figure(figsize=(16, 4))
    for index, env in enumerate(ENVIRONMENTS):
        ax = fig.add_subplot(1, 3, index+1, projection='3d' if env != 'pusht' else None)
        rows = [r for r in trajectories if r['environment'] == env and r['case'] == 0]
        for r in rows:
            p = np.asarray(r['positions'])
            if p.ndim == 2 and p.shape[1] >= 2:
                style = dict(STYLES[r['mode']], linewidth=1.5)
                if 'marker' in style:
                    style.update(markevery=max(1, len(p)//16), markersize=3)
                if env != 'pusht' and p.shape[1] == 3:
                    ax.plot(p[:, 0], p[:, 1], p[:, 2], label=r['mode'], **style)
                elif env == 'pusht':
                    ax.plot(p[:, 0], p[:, 1], label=r['mode'], **style)
        ax.set_title(env+' fixed case0'); ax.set_xlabel('actual X (px)' if env == 'pusht' else 'actual EEF X (m)')
        ax.set_ylabel('actual Y (px)' if env == 'pusht' else 'actual EEF Y (m)')
        if env != 'pusht':
            ax.set_zlabel('actual EEF Z (m)')
        if rows:
            ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(directory/'trajectories.png', dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True)
    p.add_argument('--skip-decode', action='store_true'); p.add_argument('--launch', action='store_true')
    a = p.parse_args()
    if not a.launch:
        print(json.dumps(dict(dry_run=True, gpu_calls=0, output=a.output, complete_stream_video_decode=not a.skip_decode)))
        return
    report(a.output, decode=not a.skip_decode)


if __name__ == '__main__':
    main()
