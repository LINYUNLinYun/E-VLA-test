"""Export actual passed commands; do not execute or invent missing evidence."""
from pathlib import Path
import json
import shlex

root=Path(__file__).resolve().parent
out=root.parents[2]/'reports/idea2_d10_dynamic_sm_20261024/reproduction_commands.md'
env='''export CUDA_DEVICE_MAX_CONNECTIONS=32
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH=.:policies/openpi/src
export OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi
export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu'''
lines=['# 实际复现命令', '', '在 `/root/nfs/embodied/benchmark` 下执行。以下来自实际执行收据；**重跑前给 `--output` 和 nsys `-o` 换一个新目录/文件名**，保留原始证据。只在已协调独占的 GPU 上运行，UUID 是本轮设备，不是永久授权。', '', '```bash', env, '```', '']
receipts={}
for path in sorted(root.glob('*.execution.json')):
    data=json.loads(path.read_text())
    for case in data.get('cases',[]):
        if case.get('status')=='passed' and 'command' in case:
            receipts[case['name']]=case
included=[]
for path in sorted(root.glob('*/result.json')):
    result=json.loads(path.read_text())
    if result.get('status')!='passed':
        continue
    name=path.parent.name
    record=receipts.get(name,{})
    command=record.get('command') or result.get('command')
    if not command:
        continue
    if isinstance(command,list):
        command=shlex.join(command)
    kind=('profile / 不作正式计时' if result.get('args',{}).get('profile') else
          '正式测量（适用范围见报告）' if result.get('performance_eligible') else '诊断或初筛')
    lines += ['## '+name, '', kind, '']
    if result.get('args',{}).get('profile') and not command.startswith('nsys '):
        lines += ['此早期结果只保存模型子命令。重新采集时在命令前加：`nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:nvtx-precapture --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop -o <新的trace路径>`。', '']
    lines += ['```bash',command,'```','']
    included.append(dict(case=name,kind=kind,result=str(path),command=command,
                         source_unchanged_receipt=record.get('runtime_sources_unchanged')))
lines += ['## CPU 离线汇总', '', '```bash',
 'policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/summarize_confirmation.py',
 "CUDA_VISIBLE_DEVICES='' policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/trace_analysis/offline_trace_matrix.py",
 "CUDA_VISIBLE_DEVICES='' policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/trace_analysis/summarize_traces.py",
 "CUDA_VISIBLE_DEVICES='' policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/trace_analysis/oldkv/analyze_oldkv.py",'```','',
 '闭环运行命令与固定案例见 `results/idea2/d10_dynamic_sm_20261024/closedloop_smoke_README.md`；已有执行目录禁止覆盖，重跑指定新的 `--output`。']
lines += ['', '## 补充公平对照分析与闭环', '', '```bash',
 "CUDA_VISIBLE_DEVICES='' policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/trace_analysis/oldkv/analyze_async_control.py",
 'policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/analyze_closedloop.py',
 'policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/analyze_fresh_quality.py',
 'policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/closedloop_smoke.py --execute --output <新的旧KV闭环目录>',
 'policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/fresh_baseline_smoke.py --execute --output <新的新KV闭环目录>',
 '```', '', '闭环默认物理GPU和EGL索引是本机本次配置；独占租约及映射必须重新核对。新KV闭环在GPU0，仅比较固定案例质量，不能与GPU1跨卡排名延迟。']
out.write_text('\n'.join(lines)+'\n')
(out.parent/'reproduction_index.json').write_text(json.dumps(included,indent=2)+'\n')
print('Exported',len(included),'passed cases to',out)
