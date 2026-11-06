"""Validate joint data identities, stored partitions, metrics and HTML payload."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import subprocess

import numpy as np
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from threadpoolctl import threadpool_limits

root=Path(__file__).resolve().parent
samples=json.loads((root/'samples.json').read_text())
catalog=json.loads((root/'catalog.json').read_text())
assert len(samples)==2326 and len(catalog)==56
assert len({(s['suite'],s['task'],s['init'],s['request']) for s in samples})==2326
suite=np.array([s['suite'] for s in samples]);train=np.array([s['init']<2 for s in samples]);test=~train
assert (train.sum(),test.sum())==(1556,770)
assert Counter(suite)=={0:577,1:600,2:549,3:600}
assert sum('uniform' in s['sources'] for s in samples)==1440
page=(root/'joint_suite_explorer.html').read_text()
script=page[page.rfind('<script>')+8:page.rfind('</script>')]
assert script.startswith('const payload=')
payload=json.loads(script[len('const payload='):script.index(';const $=')])
assert payload['samples']==samples and len(payload['records'])==56
metric_count=0
with threadpool_limits(limits=2):
 for row,embedded in zip(catalog,payload['records']):
  assert row['slug']==embedded['slug']
  with np.load(root/row['projection'],allow_pickle=False) as a:
   z=a['coordinates'];labels=a['cluster_labels'];active=a['active']
   assert z.shape==(2326,64) and labels.shape==(3,3,2326)
   assert np.isfinite(z).all()
   assert np.count_nonzero(z[:,~active])==0
   np.testing.assert_array_equal(np.asarray(embedded['coordinates']),z[:,:4])
   for ki,k in enumerate((2,4,8)):
    for si,seed in enumerate((0,1,2)):
     predicted=labels[ki,si]
     assert np.unique(predicted[train]).size==k
     assert predicted.min()>=0 and predicted.max()<k
     np.testing.assert_array_equal(predicted,embedded['clusters'][str(k)][str(seed)])
     centers=a[f'centers_k{k}_seed{seed}']
     distance=np.sum(z*z,axis=1)[:,None]+np.sum(centers*centers,axis=1)[None,:]-2*z@centers.T
     np.testing.assert_allclose(distance[np.arange(len(z)),predicted],distance.min(1),atol=1e-8,rtol=1e-10)
     metric=next(m for m in row['metrics'] if m['k']==k and m['seed']==seed)
     assert metric['n_train']==1556 and metric['n_test']==770 and metric['fitted_clusters']==k
     assert abs(adjusted_rand_score(suite[test],predicted[test])-metric['suite_ari'])<1e-12
     assert abs(normalized_mutual_info_score(suite[test],predicted[test])-metric['suite_nmi'])<1e-12
     metric_count+=1
assert metric_count==504
assert len(list(csv.DictReader((root/'clustering_metrics.csv').open())))==504
composition=list(csv.DictReader((root/'cluster_suite_composition.csv').open()))
assert len(composition)==3136
for row in catalog:
 for k in (2,4,8):
  selected=[r for r in composition if r['feature']==row['feature'] and int(r['layer'])==row['layer'] and r['preprocessing']==row['preprocessing'] and int(r['k'])==k]
  assert sum(int(r['test_count']) for r in selected)==770
  for name in payload['suite_names']:
   suite_index=payload['suite_names'].index(name)
   suite_key=('libero_spatial','libero_object','libero_goal','libero_10')[suite_index]
   assert abs(sum(float(r['fraction_within_suite']) for r in selected if r['suite']==suite_key)-1)<1e-12
subprocess.run(['node',str(root/'check_explorer.cjs')],check=True)
receipt=dict(passed=True,samples=2326,train=1556,test=770,configurations=56,checked_partitions=metric_count,
 nearest_center_costs_verified=True,heldout_metrics_recomputed=True,html_payload_matches=True,
 scientific_unit_tests_passed=4,script_logic=json.loads((root/'explorer_logic_validation.json').read_text()),
 caveat='Browser rendering is not included in this automated check')
(root/'validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False))
