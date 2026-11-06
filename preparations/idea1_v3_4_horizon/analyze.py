import csv,hashlib,io,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/idea1_v3_4/mechanism'
OUT=ROOT/'results/idea1_v3_4_horizon'
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
METRICS=('translation_l2','translation_mm_proxy','rotation_l2','gripper_abs','gripper_sign_mismatch','raw_rms6')
BLOCKS=((1,5),(6,10),(11,20),(21,30),(31,40),(41,50),(6,50),(1,50))


def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=True)
    dirs=sorted(p.parent for p in SOURCE.glob('shard*/pair*/complete.json'))
    assert len(dirs)==160
    identities=[]; values=[]; candidates=None; old_errors=[]
    for d in dirs:
        content=(d/'actions.npz').read_bytes(); identity=json.loads((d/'pair.json').read_text())
        names=None; conditions=[]
        with np.load(io.BytesIO(content),allow_pickle=False) as z:
            names=sorted(k[:-3] for k in z.files if k.endswith('_D2') and not k.startswith('full_'))
            if candidates is None:candidates=names
            assert candidates==names
            for D in (2,10):
                ref=z[f'full_D{D}'].astype(np.float64); rawref=z[f'full_D{D}_raw'].astype(np.float64)
                assert ref.shape==(50,7) and rawref.shape==(50,32)
                cond=[]
                for name in names:
                    a=z[f'{name}_D{D}'].astype(np.float64); raw=z[f'{name}_D{D}_raw'].astype(np.float64)
                    assert a.shape==ref.shape and raw.shape==rawref.shape and np.isfinite(a).all() and np.isfinite(raw).all()
                    delta=a-ref
                    cond.append(np.stack([np.linalg.norm(delta[:,:3],axis=1),
                        50*np.linalg.norm(np.clip(a[:,:3],-1,1)-np.clip(ref[:,:3],-1,1),axis=1),
                        np.linalg.norm(delta[:,3:6],axis=1),np.abs(delta[:,6]),
                        ((a[:,6]>=0)!=(ref[:,6]>=0)).astype(float),np.sqrt(np.mean((raw[:,:6]-rawref[:,:6])**2,axis=1))],axis=-1))
                conditions.append(cond)
        values.append(conditions)
        with (d/'action_metrics.csv').open() as f: legacy=list(csv.DictReader(f))
        for r in legacy:
            if r['component']!='translation':continue
            computed=np.mean(values[-1][(2,10).index(int(r['D']))][names.index(r['candidate'])][:int(r['horizon']),1])
            old_errors.append(abs(computed-float(r['clipped_controller_translation_mm_proxy'])))
        identities.append(dict(path=str(d.relative_to(ROOT)),episode=identity['episode'],case=identity['case'],actions_sha256=hashlib.sha256(content).hexdigest()))
    # D, independent saved request, candidate, prediction index, metric.
    data=np.moveaxis(np.asarray(values,dtype=np.float64),1,0)
    assert data.shape==(2,160,len(candidates),50,len(METRICS)) and np.isfinite(data).all()
    assert max(old_errors)<1e-4
    np.savez_compressed(OUT/'per_request_errors.npz',errors=data,candidates=np.asarray(candidates),metrics=np.asarray(METRICS),D=np.asarray([2,10]),steps=np.arange(1,51))
    selected=candidates.index('v14'); intervals={}; paired_intervals={}; rng=np.random.default_rng(291029)
    # Shared bootstrap draws preserve candidate pairing; resample suite-stratified tasks.
    weights={s:rng.multinomial(10,np.ones(10)/10,size=2000)/10 for s in SUITES}
    for di,D in enumerate((2,10)):
        boot=np.zeros((2000,len(candidates),50),float)
        for s in SUITES:
            groups=[]
            for t in range(10):
                idx=[i for i,m in enumerate(identities) if m['case']['suite']==s and m['case']['task']==t]
                assert len(idx)==4
                groups.append(data[di,idx,:,:,1].mean(axis=0))
            boot+=np.einsum('bt,tch->bch',weights[s],np.asarray(groups))/4
        intervals[D]=np.percentile(boot,[2.5,97.5],axis=0)
        paired_intervals[D]=np.percentile(boot-boot[:,selected:selected+1,:],[2.5,97.5],axis=0)
        np.savez_compressed(OUT/f'translation_bootstrap_D{D}.npz',means=boot)
    point=[]; prefix=[]; segments=[]; effects=[]
    for di,D in enumerate((2,10)):
        for scope in ('all',*SUITES):
            ix=[i for i,m in enumerate(identities) if scope=='all' or m['case']['suite']==scope]
            a=data[di,ix]
            for ci,name in enumerate(candidates):
                for mi,metric in enumerate(METRICS):
                    x=a[:,ci,:,mi]; cum=np.cumsum(x,axis=1)/np.arange(1,51)
                    for j in range(50):
                        stats=dict(D=D,suite=scope,candidate=name,metric=metric,step=j+1,n=len(ix))
                        p=dict(stats,mean=float(x[:,j].mean()),p50=float(np.median(x[:,j])),p90=float(np.percentile(x[:,j],90)))
                        if scope=='all' and metric=='translation_mm_proxy':p.update(ci_low=float(intervals[D][0,ci,j]),ci_high=float(intervals[D][1,ci,j]))
                        point.append(p)
                        prefix.append(dict(stats,mean=float(cum[:,j].mean()),p50=float(np.median(cum[:,j])),p90=float(np.percentile(cum[:,j],90))))
                    for lo,hi in BLOCKS:
                        block=x[:,lo-1:hi].mean(axis=1)
                        segments.append(dict(D=D,suite=scope,candidate=name,metric=metric,start=lo,end=hi,n=len(ix),mean=float(block.mean()),p50=float(np.median(block)),p90=float(np.percentile(block,90))))
                for j in range(50):
                    delta=a[:,ci,j,1]-a[:,selected,j,1]
                    row=dict(D=D,suite=scope,candidate=name,step=j+1,n=len(ix),delta_mm=float(delta.mean()),improved_fraction=float((delta<0).mean()))
                    if scope=='all':row.update(ci_low=float(paired_intervals[D][0,ci,j]),ci_high=float(paired_intervals[D][1,ci,j]))
                    effects.append(row)
    # Optional CI columns use a stable union for CSV headers.
    def union_write(path,rows):
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    union_write(OUT/'per_step.csv',point);write(OUT/'cumulative_prefix.csv',prefix);write(OUT/'segments.csv',segments);union_write(OUT/'paired_vs_v14_per_step.csv',effects)
    manifest=dict(time=time.time(),source='Frozen v3.4 E1 160 same-input/same-generation-noise pairs; cache age5 executed steps',
        cases=40,requests=160,D=[2,10],H=50,R=5,candidates=candidates,metrics=METRICS,identities=identities,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),legacy_h5_h50_max_abs_difference=max(old_errors),bootstrap=dict(draws=2000,seed=291029,unit='task',stratify='suite',pointwise_not_simultaneous=True))
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    plots(data,candidates)
    report(data,candidates,segments,paired_intervals)
    # Independently reconcile interval means with per-step means and cumulative endpoints.
    mean=data.mean(axis=1)
    for di,D in enumerate((2,10)):
        for ci,name in enumerate(candidates):
            for lo,hi in BLOCKS:
                for mi,metric in enumerate(METRICS):
                    actual=next(r['mean'] for r in segments if r['D']==D and r['suite']=='all' and r['candidate']==name and r['metric']==metric and r['start']==lo and r['end']==hi)
                    assert np.isclose(actual,mean[di,ci,lo-1:hi,mi].mean(),rtol=1e-12,atol=1e-12)
    (OUT/'verification.json').write_text(json.dumps(dict(passed=True,requests=160,candidates=len(candidates),steps=50,D=[2,10],legacy_h5_h50_max_abs_difference=max(old_errors),finite=True,per_step_segment_reconciliation=True,wall_s=time.time()-started),indent=2)+'\n')
    print(json.dumps(dict(requests=160,candidates=len(candidates),passed=True,output=str(OUT),wall_s=time.time()-started)))


def plots(data,candidates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['v14','old_suffix_L16','old_suffix_L17','all_old','oracle_suffix_L16']
    colors=['#333333','#d95f02','#1b9e77','#7570b3','#e7298a'];t=np.arange(1,51)
    for di,D in enumerate((2,10)):
        fig,axs=plt.subplots(2,2,figsize=(12,7))
        for name,col in zip(names,colors):
            ci=candidates.index(name);a=data[di,:,ci];avg=a.mean(axis=0)
            axs[0,0].plot(t,avg[:,1],label=name,color=col)
            axs[0,1].plot(t,np.cumsum(avg[:,1])/t,label=name,color=col)
            axs[1,0].plot(t,avg[:,2],label=name,color=col)
            axs[1,1].plot(t,avg[:,4]*100,label=name,color=col)
        for ax in axs.flat:ax.axvline(5.5,color='grey',linestyle='--');ax.set_xlabel('Prediction index within one H50 chunk');ax.grid(alpha=.2)
        axs[0,0].set_ylabel('Translation command difference proxy (mm)')
        axs[0,1].set_ylabel('Cumulative prefix mean proxy (mm)')
        axs[1,0].set_ylabel('Rotation command L2 difference')
        axs[1,1].set_ylabel('Gripper sign disagreement (%)')
        fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,fontsize=9)
        fig.suptitle(f'D{D}: same-observation action differences, R=5 (dashed); later predictions unexecuted',y=.92,fontsize=10)
        fig.tight_layout(rect=(0,0,1,.90));fig.savefig(OUT/f'horizon_D{D}.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    base=candidates.index('v14')
    for di,D in enumerate((2,10)):
        delta=np.asarray([data[di,:,candidates.index(f'old_single_L{layer}'),:,1].mean(axis=0)-data[di,:,base,:,1].mean(axis=0) for layer in range(1,19)])
        lim=max(abs(delta.min()),abs(delta.max()));im=axs[di].imshow(delta,aspect='auto',origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim,extent=(.5,50.5,.5,18.5))
        axs[di].set_title(f'D{D}: single-layer old KV minus V14');axs[di].set_xlabel('Prediction index');axs[di].set_ylabel('Replaced VLM layer');axs[di].axvline(5.5,color='black',linestyle='--');fig.colorbar(im,ax=axs[di],label='Translation proxy change (mm)')
    fig.tight_layout();fig.savefig(OUT/'single_layer_horizon.png',dpi=180);plt.close(fig)


def report(data,candidates,segments,paired_intervals):
    def cell(name,D,metric,lo,hi,suite='all'):
        return next(r['mean'] for r in segments if (r['candidate'],r['D'],r['metric'],r['start'],r['end'],r['suite'])==(name,D,metric,lo,hi,suite))
    names=['v14','old_suffix_L12','old_suffix_L13','old_suffix_L15','old_suffix_L16','old_suffix_L17','old_suffix_L18','all_old','oracle_suffix_L16']
    lines=['# v3.4 E1：完整H50预测误差重新统计','',
        '复用原160对同观测、同生成噪声的存档；四套件各10task，每task一条init14 Full轨迹、4个预选请求。D2/D10均为相同160对。CPU读取完整raw/decoded动作重算，无新推理或闭环，无修改旧结果。',
        '', '**这里是一次预测chunk内第1–50步相对同输入Full的动作差，不是未来实际EEF轨迹误差。实际R=5，只执行前五步；第6–50步是未执行的预测后缀。此bank缓存年龄固定为5个已执行控制步，不是age10，也不能直接当成B/C自由闭环轨迹的误差。**']
    lines += ['', '## 主要发现', '',
        '前五步的结论不能推广到整个H50。D2中，复用L16–18前五步平均比V14高0.329mm，但第6–50步低2.051mm（27.8%）；复用L17–18前五步低15.3%，第6–50步低28.1%。D10后段分别低29.3%和29.9%，方向一致。',
        '', '曲线也不是单调增加：V14从第6–10步的6.293mm升到第41–50步的7.834mm；两种尾层复用先降到约4.8–5.0mm，再回到约5.4–5.6mm。L16–18与L17–18的第6–50步均值已很接近，分别为5.337和5.313mm。',
        '', '单层L16也呈现前后段反转：前五步比V14高0.874mm，第6–50步低0.240mm；单层L17后段改善更大（低1.976mm）。因此“L16总是有害”同样需要限定为这里的近端预测表现。',
        '', '后段平移差降低在四个套件中都能看到，旋转差也降低；这不是仅由平移clip指标造成的，未clip平移和raw前六维误差方向一致。夹爪则需单独看：L16–18前五步符号不一致率6.50%，高于V14的3.38%，后段降为2.26%，低于V14的2.79%。',
        '', '这些结果说明误差集中在哪些预测位置，尚不能识别其成因。后段未执行，不能用H50平均误差替代前五步质量或闭环成功率，也不能据此直接认定延长执行窗口会改善成功率。',
        '', '## 相对V14的配对分段平移差', '',
        '负值表示更接近同输入Full；95%区间沿用suite分层、task聚类bootstrap。以下为探索性分段比较，未校正多重比较。', '',
        '|D|条件|预测段|均值差 mm|95%区间 mm|', '|---|---|---|---|---|']
    for di,D in enumerate((2,10)):
        boot=np.load(OUT/f'translation_bootstrap_D{D}.npz')['means']
        bi=candidates.index('v14')
        for name in ('old_suffix_L16','old_suffix_L17'):
            ci=candidates.index(name)
            for lo,hi in ((1,5),(6,50),(41,50),(1,50)):
                delta=cell(name,D,'translation_mm_proxy',lo,hi)-cell('v14',D,'translation_mm_proxy',lo,hi)
                low,high=np.percentile((boot[:,ci,lo-1:hi]-boot[:,bi,lo-1:hi]).mean(axis=1),[2.5,97.5])
                lines.append(f'|{D}|{name}|{lo}–{hi}|{delta:+.3f}|[{low:+.3f}, {high:+.3f}]|')
    for D in (2,10):
        lines += ['',f'## D{D} 平移控制指令误差代理（mm）','', '|条件|1–5|6–10|11–20|21–30|31–40|41–50|6–50|1–50|','|---|---|---|---|---|---|---|---|---|']
        for name in names:lines.append('|'+ '|'.join([name]+[f'{cell(name,D,"translation_mm_proxy",lo,hi):.3f}' for lo,hi in BLOCKS])+'|')
        lines += ['',f'![D{D}逐预测步、累计前缀、旋转与夹爪](../../../results/idea1_v3_4_horizon/horizon_D{D}.png)']
    lines += ['', '## D2 分套件：前段与后段','', '|suite|条件|1–5 mm|6–50 mm|41–50 mm|','|---|---|---|---|---|']
    for suite in SUITES:
        for name in ('v14','old_suffix_L16','old_suffix_L17'):
            lines.append('|'+ '|'.join([suite,name]+[f'{cell(name,2,"translation_mm_proxy",lo,hi,suite):.3f}' for lo,hi in ((1,5),(6,50),(41,50))])+'|')
    lines += ['', '## 单层复用：D2前后段平移对比', '',
        '|条件|1–5 mm|6–50 mm|41–50 mm|', '|---|---|---|---|']
    for name in ('v14','old_single_L15','old_single_L16','old_single_L17','old_single_L18'):
        lines.append('|'+ '|'.join([name]+[f'{cell(name,2,"translation_mm_proxy",lo,hi):.3f}' for lo,hi in ((1,5),(6,50),(41,50))])+'|')
    lines += ['', '## 旋转与夹爪分段（D2）','', '|条件|旋转L2 1–5|旋转L2 6–50|夹爪符号差1–5|夹爪符号差6–50|','|---|---|---|---|---|']
    for name in names:
        lines.append(f'|{name}|{cell(name,2,"rotation_l2",1,5):.4f}|{cell(name,2,"rotation_l2",6,50):.4f}|{cell(name,2,"gripper_sign_mismatch",1,5):.2%}|{cell(name,2,"gripper_sign_mismatch",6,50):.2%}|')
    lines += ['', '![单层复用在整个H50的影响](../../../results/idea1_v3_4_horizon/single_layer_horizon.png)',
        '', '平移mm代理=clip后平移控制命令差的L2×50mm控制缩放；rotation为decoded旋转命令坐标L2，不是SO(3)角距离。逐点CSV包含均值/中位数/P90；累计前缀先对每请求的前h步平均，再跨请求聚合。raw_rms6排除夹爪和padding维。',
        '', 'translation_mm_proxy逐点区间及相对V14的配对区间按suite分层task聚类bootstrap2000次；为逐点探索区间，非50步同时置信带，未作多重比较校正。单层/后缀最优均为描述性，不据此重选闭环策略。',
        '', '[逐点结果](../../../results/idea1_v3_4_horizon/per_step.csv) · [累计前缀](../../../results/idea1_v3_4_horizon/cumulative_prefix.csv) · [分段结果](../../../results/idea1_v3_4_horizon/segments.csv) · [相对V14逐点配对差](../../../results/idea1_v3_4_horizon/paired_vs_v14_per_step.csv) · [重算核对](../../../results/idea1_v3_4_horizon/verification.json)']
    (ROOT/'docs/experiments/idea1/idea1_v3_4_horizon_report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
