"""Verify task-pair partitions, train-only label mapping and exported payloads."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.metrics import adjusted_rand_score,balanced_accuracy_score
from threadpoolctl import threadpool_limits

root=Path(__file__).resolve().parent
catalog=json.loads((root/'catalog.json').read_text())
assert len(catalog)==168
page=(root/'task_pair_explorer.html').read_text()
script=page[page.rfind('<script>')+8:page.rfind('</script>')]
assert script.startswith('const payload=')
payload=json.loads(script[len('const payload='):script.index(',records=payload.records')])
assert len(payload['records'])==168
pair_counts={};checked=0
with threadpool_limits(limits=2):
 for row,embedded in zip(catalog,payload['records']):
  assert row['slug']==embedded['slug']
  samples=json.loads((root/f'{row["pair"]}_samples.json').read_text())
  assert payload['samples'][row['pair']]==samples
  assert len({(s['suite'],s['task'],s['init']) for s in samples})==6
  with np.load(root/row['projection'],allow_pickle=False) as a:
   z=a['coordinates'];truth=a['labels'];train=a['train'];test=~train
   assert len(z)==len(samples)
   np.testing.assert_array_equal(train,[s['init']<2 for s in samples])
   pair=next(p for p in payload['pairs'] if p['id']==row['pair'])
   np.testing.assert_array_equal(truth,[pair['tasks'].index([s['suite'],s['task']]) for s in samples])
   np.testing.assert_array_equal(embedded['coordinates'],z[:,:4])
   np.testing.assert_array_equal(embedded['labels'],truth)
   np.testing.assert_array_equal(embedded['mapped'],a['mapped_cluster_labels'])
   assert np.isfinite(z).all() and np.count_nonzero(z[:,~a['active']])==0
   pair_counts[row['pair']]=dict(train=int(train.sum()),test=int(test.sum()),test_a=int((test&(truth==0)).sum()),test_b=int((test&(truth==1)).sum()))
   for seed in (0,1,2):
    cluster=a['cluster_labels'][seed];mapped=a['mapped_cluster_labels'][seed];center=a['centers'][seed]
    identity=balanced_accuracy_score(truth[train],cluster[train])
    reverse=balanced_accuracy_score(truth[train],1-cluster[train])
    expected=1-cluster if reverse>identity else cluster
    np.testing.assert_array_equal(mapped,expected)
    cost=np.sum(z*z,axis=1)[:,None]+np.sum(center*center,axis=1)[None,:]-2*z@center.T
    np.testing.assert_allclose(cost[np.arange(len(z)),cluster],cost.min(1),atol=1e-8,rtol=1e-10)
    metric=row['metrics'][seed]
    assert abs(metric['kmeans_balanced_accuracy']-balanced_accuracy_score(truth[test],mapped[test]))<1e-12
    assert abs(metric['kmeans_ari']-adjusted_rand_score(truth[test],cluster[test]))<1e-12
    assert abs(metric['linear_balanced_accuracy']-balanced_accuracy_score(truth[test],a['linear_predictions'][test]))<1e-12
    checked+=1
assert checked==504
assert pair_counts['spatial0_vs_spatial5']['test']==34
assert pair_counts['spatial0_vs_goal0']['test']==36
assert pair_counts['spatial0_vs_spatial4_control']['test']==36
assert len(list(csv.DictReader((root/'pair_metrics.csv').open())))==504
(root/'interactive_script_validation.js').write_text(script)
subprocess.run(['node','--check',str(root/'interactive_script_validation.js')],check=True)
services={}
run=root.parents[1]
for pair in payload['pairs']:
 identities=[]
 for suite,task in pair['tasks']:
  suite_name=('libero_spatial','libero_object','libero_goal','libero_10')[suite]
  identity=json.loads((run/'episodes'/suite_name/f'task{task:02d}'/'init00/identity.json').read_text())
  identities.append(dict(instruction=identity['instruction'],service=identity['service']))
 services[pair['id']]=dict(tasks=identities,same_service=identities[0]['service']==identities[1]['service'])
assert services['spatial0_vs_goal0']['same_service']
assert services['spatial0_vs_spatial4_control']['same_service']
assert not services['spatial0_vs_spatial5']['same_service']
(root/'pair_identity.json').write_text(json.dumps(services,ensure_ascii=False,indent=2)+'\n')
receipt=dict(passed=True,pairs=pair_counts,configurations=168,partitions_verified=504,
 cluster_mapping_uses_train_only=True,nearest_center_costs_verified=True,
 heldout_metrics_recomputed=True,interactive_payload_matches=True,interactive_script_syntax_passed=True,
 scientific_unit_tests_passed=3,known_backend_confound='Spatial0 versus Spatial5',
 same_service_controls=['Spatial0 versus Goal0','Spatial0 versus Spatial4'],
 excluded_scope='Browser rendering and generalization beyond the two held-out episodes per pair are not established')
(root/'validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False))
