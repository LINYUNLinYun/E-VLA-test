import json,sqlite3
from pathlib import Path
from collections import defaultdict
from bisect import bisect_right
root=Path('results/idea2/cewp_20261021')

def ae_kernels(path,section_prefix):
 db=sqlite3.connect(path)
 ranges=db.execute('SELECT n.start,n.end,n.globalTid,COALESCE(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL ORDER BY n.start').fetchall()
 sections=[r for r in ranges if r[3] and r[3].startswith(section_prefix)];starts=[r[0] for r in sections]
 aes=defaultdict(list)
 for b,e,tid,l in ranges:
  if l=='ae':aes[tid].append((b,e))
 query='''SELECT k.start,k.end,r.start,r.end,r.globalTid,s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId AND k.globalPid=(r.globalTid & -16777216) JOIN StringIds s ON k.demangledName=s.id ORDER BY r.start'''
 result=[]
 for b,e,launch,lend,tid,name,*geom in db.execute(query):
  si=bisect_right(starts,launch)-1
  if si<0 or lend>sections[si][1]:continue
  if not any(mb<=launch and lend<=me for mb,me in aes[tid]):continue
  result.append(dict(name=name,geometry=geom,us=(e-b)/1000))
 return result
qkv=ae_kernels('nsys/idea2_experiment/cewp_stages_pi05.1.sqlite','stage/qkv__qkv/serial')
qkv_sigs={(x['name'],tuple(x['geometry'])) for x in qkv}
results={}
for number,mode in [(3,'graph_serial'),(4,'graph_ae_high'),(5,'graph_pre_high')]:
 rows=ae_kernels(f'nsys/idea2_experiment/cewp_d10_full.{number}.sqlite','cewp/continuous/')
 total=sum(x['us'] for x in rows);groups={}
 predicates={
 'cat_all':lambda r:'CatArrayBatchedCopy' in r['name'],
 'direct_copy_all_not_all_are_KV':lambda r:'direct_copy_kernel_cuda' in r['name'],
 'bfloat16_cast_copy':lambda r:'bfloat16_copy_kernel_cuda' in r['name'],
 'cat_or_named_copy':lambda r:any(s in r['name'] for s in ('CatArrayBatchedCopy','direct_copy_kernel_cuda','bfloat16_copy_kernel_cuda')),
 'matrix_gemm_gemv':lambda r:'gemm' in r['name'].lower() or 'gemv' in r['name'].lower(),
 'matrix_splitk_epilogue_aux':lambda r:'splitKreduce' in r['name'] or 'epilogue' in r['name'],
 'qkv_matching_measured_phase_signature':lambda r:(r['name'],tuple(r['geometry'])) in qkv_sigs,
 'kernel_duration_le_5us':lambda r:r['us']<=5,
 'kernel_duration_le_2us':lambda r:r['us']<=2,
 }
 for label,predicate in predicates.items():
  subset=[r for r in rows if predicate(r)];busy=sum(x['us'] for x in subset)
  groups[label]=dict(calls=len(subset),calls_fraction=len(subset)/len(rows),total_ms=busy/1000,ms_per_request=busy/5000,kernel_time_fraction=busy/total)
 results[mode]=dict(requests=5,ae_kernel_calls=len(rows),ae_sum_kernel_ms=total/1000,ae_ms_per_request=total/5000,categories=groups)
result=dict(modes=results,qkv_signature_source='nsys/idea2_experiment/cewp_stages_pi05.1.sqlite',qkv_signature_types=len(qkv_sigs),caveats=['Category sets overlap; do not add their percentages.','Cat includes RoPE and metadata/suffix concatenation, not only prefix KV. Named copy includes precision conversion and temporary copies, not only KV.','QKV signature match is based on real layer0 QKV stage; identical kernel name/launch geometry is supporting attribution, not a full operator-level unique identifier.','Timing is summed profiler kernel elapsed intervals, excluding host gaps and DMA memcpy. Concurrent intervals include stalls.','The 37 FP32 GEMV calls per denoising step match 18*2+1 adaptive RMSNorm dense(cond) sites; mapping requires operator hooks before optimizing.'])
(root/'ae_kernel_mix.json').write_text(json.dumps(result,indent=2)+'\n')
for mode,r in results.items():
 print(mode,'AE ms',r['ae_ms_per_request'])
 for k,v in r['categories'].items():print(k,round(v['ms_per_request'],4),round(v['kernel_time_fraction']*100,2),'%',v['calls'])
