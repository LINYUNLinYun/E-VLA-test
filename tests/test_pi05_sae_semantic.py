"""Scientific invariants for the SAE pilot; no model weights or GPU required."""
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import torch
import pytest

from experiments.pi05_sae_semantic.native_probe import NativeProbe
from experiments.pi05_sae_semantic.map_concept_ffn import semantic_change
from experiments.pi05_sae_semantic.map_concept_ffn import calibration_entry_supported
from experiments.pi05_sae_semantic import map_concept_ffn
from experiments.pi05_sae_semantic.collect_local import bounded_observations
from experiments.pi05_sae_semantic.build_semantic_rules import propose_masks
from experiments.pi05_sae_semantic.runtime_router import select_mask
from experiments.pi05_sae_semantic.native_probe import sampling_layout
from experiments.pi05_sae_semantic.select_sae_concepts import balanced_weights, concept_label, weighted_quantile


def test_task_episode_weights_ignore_trajectory_length():
    rows = [dict(suite="s", task=0, init=0)] + [dict(suite="s", task=0, init=1)] * 3 + [dict(suite="s", task=1, init=0)] * 2
    weights = balanced_weights(rows)
    assert np.isclose(weights[:4].sum(), .5)
    assert np.isclose(weights[4:].sum(), .5)
    assert np.isclose(weights[0], .25)
    assert np.isclose(weights[1:4].sum(), .25)


def test_expansion_includes_pilot_endpoints_and_all_episodes():
    rows=[dict(suite='s',task=task,init=init,request=frame) for task in range(16) for init in (0,1) for frame in range(24)]
    selected=bounded_observations(rows,128)
    assert len({(r['task'],r['init']) for r in selected})==32
    assert len({(r['task'],r['init'],r['request']) for r in selected})==128
    for task in range(16):
        for init in (0,1):
            assert {0,23} <= {r['request'] for r in selected if r['task']==task and r['init']==init}


def test_pair_proposals_respect_adjacency_and_missing_statistics():
    modules=[6,8,10,12,14,15]
    rows=[dict(mask=str(m),concept=c,D=str(value),supported='True')
          for c,values in [('a',[1,2,3,4,5,6]),('b',[6,5,4,3,2,1])] for m,value in zip(modules,values)]
    selected,ranked=propose_masks(rows,3)
    assert 1<=len(selected)<=3
    assert all(p['mask']!=[14,15] for p in ranked)
    assert all(len(set(p['mask']))==2 for p in selected)
    # Missing statistics cannot masquerade as zero impact and make a pair win.
    for r in rows:
        if r['mask']=='6':r.update(D='',supported='False')
    selected,ranked=propose_masks(rows,3)
    assert all(6 not in p['mask'] for p in ranked)
    identical=[dict(mask=str(m),concept=c,D=str(value),supported='True') for c in ('a','b') for m,value in zip(modules,[1,2,3,4,5,6])]
    selected,ranked=propose_masks(identical,3,diversify=True)
    assert len(selected)==3
    assert any(6 not in p['mask'] for p in selected)
    assert any(8 not in p['mask'] for p in selected)


def test_turn_on_is_not_a_spatial_label():
    assert not concept_label("turn on the stove", "spatial", "on")
    assert concept_label("turn on the stove and put the pot on it", "spatial", "on")
    assert concept_label("put the bowl on the plate", "object", "bowl")


def test_online_rule_protects_all_active_concepts():
    rules=dict(delta_semantic=1.,early_concepts={c:dict(theta=.8) for c in ('bowl','on','stove')},
               masks=[dict(mask=[6,15],supported_concepts={'bowl':{'D':.5},'on':{'D':.9}})])
    assert select_mask({'bowl':.9,'on':.9,'stove':0.},rules)[0]==(6,15)
    assert select_mask({'bowl':.9,'on':.9,'stove':.9},rules)[0]==()
    assert select_mask({'bowl':0.,'on':0.,'stove':0.},rules)[0]==()


def test_shared_sampling_excludes_padding_and_inactive_camera():
    class Tokenizer:
        def encode(self,text,add_bos=False):return [1,2,3] if add_bos else [4]
    p=SimpleNamespace(images=[None]*3,image_masks=[torch.tensor([True]),torch.tensor([True]),torch.tensor([False])],
                      tokens=torch.tensor([[1,2,3,4,0,0]]),token_masks=torch.tensor([[True,True,True,True,False,False]]))
    layout=sampling_layout(p,'a task',3*256+6,Tokenizer())
    assert len(layout['selected_positions'])==64
    assert layout['selected_positions'][:2]==[769,770]
    assert all(position<512 or position in (769,770) for position in layout['selected_positions'])


def test_intervention_metric_sees_saturated_changes_without_cancellation():
    features = [dict(tau=1.,b=1.,weight=1.)]
    assert semantic_change([10.],[3.],features) == (7.,0.)
    features = [dict(tau=0.,b=1.,weight=.5)]*2
    assert semantic_change([1.,0.],[0.,1.],features) == (1.,1.)


def test_routing_support_is_conditional_calibration_not_global_late_classifier():
    assert calibration_entry_supported('calibration',(6,8),True,True,2)
    assert not calibration_entry_supported('discovery',(6,8),True,True,2)
    assert not calibration_entry_supported('calibration',(6,),True,True,2)
    assert not calibration_entry_supported('calibration',(6,8),True,True,0)
    assert not calibration_entry_supported('calibration',(6,8),True,False,2)


def test_ffn_switch_bypasses_linear_and_restores_it(monkeypatch):
    probe = object.__new__(NativeProbe)
    probe.lm = SimpleNamespace(layers=[SimpleNamespace(mlp=torch.nn.Linear(2, 2)) for _ in range(18)])
    layer = probe.lm.layers[8].mlp
    x = torch.tensor([[1., 2.]])
    expected = layer(x).detach().clone()
    original = torch.nn.functional.linear
    with probe.skip_ffn((8,)):
        def forbidden(*args, **kwargs):
            raise AssertionError("Skipped FFN must not evaluate its linear projection")
        monkeypatch.setattr(torch.nn.functional, "linear", forbidden)
        assert torch.equal(layer(x), torch.zeros_like(x))
    monkeypatch.setattr(torch.nn.functional, "linear", original)
    assert torch.equal(layer(x), expected)


@pytest.mark.parametrize('split_features',[False,True])
def test_rare_precise_features_survive_full_selection(tmp_path,split_features):
    collection, assets, output = tmp_path/"collection", tmp_path/"assets", tmp_path/"output"
    (assets/"concepts").mkdir(parents=True)
    index = {f"paligemma_layer_{layer}": {"object": {"bowl":{"feature_indices":[1,4] if split_features else [1]}, "stove":{"feature_indices":[2]}},
                                           "spatial":{"on":{"feature_indices":[3]}}} for layer in (5,16)}
    (assets/"concepts/concept_features_pi05.json").write_text(json.dumps(index))
    instructions = ["put the bowl on the plate", "put the bowl on the cabinet", "turn on the stove", "push the plate to the front of the stove"]
    identities = []
    for split in ("discovery", "calibration"):
        for task, text in enumerate(instructions):
            for init in range(2):
                for frame in range(5):
                    identity = f"{split}_{task}_{init}_{frame}"
                    identities.append(identity)
                    folder = collection/"observations"/identity
                    folder.mkdir(parents=True)
                    row = dict(id=identity, split=split, suite=split, task=task, init=init, instruction=text)
                    (folder/"complete.json").write_text(json.dumps(row))
                    # Exactly 30% positive coverage; negative activations remain zero.
                    active = frame == 0 or (frame == 1 and init == 0)
                    a = np.array([10.*active*(task<2), 10.*active*(task>=2), 10.*active*(task<2)], dtype=np.float32)
                    if split_features:
                        alternative=frame==4 or (frame==3 and init==1)
                        a=np.r_[a,np.float32(10.*alternative*(task<2))]
                    np.savez(folder/"readouts.npz", **{f"a{layer}_{kind}":a for layer in (5,16) for kind in ("all","image","instruction")})
    ids=[1,2,3,4] if split_features else [1,2,3]
    (collection/"collection_config.json").write_text(json.dumps(dict(observations=identities,feature_ids={"5":ids,"16":ids})))
    command = [sys.executable, "-m", "experiments.pi05_sae_semantic.select_sae_concepts", "--collection", str(collection),
               "--assets", str(assets), "--output", str(output)]
    run = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    result = json.loads((output/"concept_features.json").read_text())
    for item in result.values():
        assert item["ready_for_intervention"]
        for layer in ("5", "16"):
            feature = item["layers"][layer]["features"][0]
            assert np.isclose(feature["positive_frequency"], .3)
            assert feature["tau"] == 0 and feature["r"] == 10 and feature["b"] == 10
    if split_features:
        for layer in ('5','16'):
            item=result['object/bowl']['layers'][layer]
            assert len(item['features'])==1
            assert item['feature_count_options'][1]['triggered']==0
    assert weighted_quantile(np.array([0.] * 14 + [10.] * 6), np.ones(20), .5) == 0


def test_matrix_cohort_uses_l5_even_when_l16_and_labels_do_not_support_it(tmp_path, monkeypatch):
    collection,selection,output=tmp_path/'collection',tmp_path/'selection',tmp_path/'output'
    collection.mkdir();selection.mkdir()
    checkpoint=str((tmp_path/'checkpoint').resolve())
    (collection/'collection_config.json').write_text(json.dumps(dict(device='cpu',checkpoint=checkpoint,feature_ids={'16':[0]})))
    (selection/'semantic_selection_summary.json').write_text(json.dumps(dict(partial=False,ready_concepts=['object/bowl'])))
    feature=dict(feature_id=0,tau=0.,b=1.,weight=1.,discovery_metrics=dict(positive_tasks=2))
    definition={'object/bowl':dict(category='object',concept='bowl',layers={'5':dict(theta=.8,supported=True),'16':dict(features=[feature],supported=True)})}
    (selection/'concept_features.json').write_text(json.dumps(definition))
    # Deliberately no L16 concept-score array: it must not censor the cohort.
    np.savez(selection/'semantic_scores.npz',L5_object_bowl_all=np.ones(2))
    rows=[]
    for i in range(2):
        identity=f'case{i}';folder=collection/'observations'/identity;folder.mkdir(parents=True)
        rows.append(dict(id=identity,split='discovery',suite='s',task=i,init=0,instruction='pick the item in the basket',input_sha256='fixture'))
        np.savez(folder/'readouts.npz',a16_all=np.zeros(1),a16_instruction=np.zeros(1))
        torch.save(dict(hidden5=torch.zeros(1,2,1),attention=torch.zeros(1),positions=torch.zeros(1),selected=[0,1],kinds=['instruction','camera_0']),folder/'prefix_tensors.pt')
    (selection/'score_observations.json').write_text(json.dumps(rows))
    class FakeProbe:
        def __init__(self,*args,**kwargs):self.backend=SimpleNamespace(device=torch.device('cpu'))
        def tail(self,*args,**kwargs):return torch.tensor([[[2.],[3.]]])
    monkeypatch.setattr(map_concept_ffn,'NativeProbe',FakeProbe)
    monkeypatch.setattr(map_concept_ffn,'load_official_loader',lambda p:SimpleNamespace(load_sae=lambda *a,**k:SimpleNamespace(encode=lambda x:x)))
    monkeypatch.setattr(sys,'argv',['map_concept_ffn','--collection',str(collection),'--selection',str(selection),
                                 '--assets',str(tmp_path),'--output',str(output),'--checkpoint',checkpoint,'--max-observations','2'])
    map_concept_ffn.main()
    result=json.loads((output/'matrix_status.json').read_text())
    assert result['intervention_forwards']==12
    assert all(r['observations']==2 and r['supported'] and r['D']==3 and r['E']==1 for r in result['matrix'])
    assert all(not r['routing_supported'] for r in result['matrix'])
    # Re-score the same physical interventions using instruction tokens only.
    # Cached maxima must be sufficient: a second model must never be loaded.
    second_output=tmp_path/'instruction_output'
    (selection/'semantic_selection_summary.json').write_text(json.dumps(dict(partial=False,ready_concepts=['object/bowl'],token_pool='instruction')))
    np.savez(selection/'semantic_scores.npz',L5_object_bowl_instruction=np.ones(2))
    def forbidden_model(*args,**kwargs):raise AssertionError('Forward results must be reused between token-pool analyses')
    monkeypatch.setattr(map_concept_ffn,'NativeProbe',forbidden_model)
    monkeypatch.setattr(sys,'argv',['map_concept_ffn','--collection',str(collection),'--selection',str(selection),
                                 '--assets',str(tmp_path),'--output',str(second_output),'--checkpoint',checkpoint,'--max-observations','2'])
    map_concept_ffn.main()
    second=json.loads((second_output/'matrix_status.json').read_text())
    assert second['intervention_forwards']==0 and second['reused_forward_results']==12
    assert all(r['D']==2 and r['E']==1 for r in second['matrix'])
    # Failed late semantic calibration is visible in exploratory analysis and
    # can never silently become a validated routing-table entry.
    definition['object/bowl']['layers']['16']['supported']=False
    (selection/'concept_features.json').write_text(json.dumps(definition))
    exploratory=tmp_path/'exploratory_output'
    monkeypatch.setattr(sys,'argv',['map_concept_ffn','--collection',str(collection),'--selection',str(selection),
        '--assets',str(tmp_path),'--output',str(exploratory),'--checkpoint',checkpoint,'--max-observations','2','--evidence-level','l5_calibrated'])
    map_concept_ffn.main()
    exploratory_result=json.loads((exploratory/'matrix_status.json').read_text())
    assert all(not r['late_semantic_calibrated'] and not r['routing_supported'] for r in exploratory_result['matrix'])
    # Raw forward caches remain reusable, but old D/E measurements may not be
    # silently reused after changing feature normalization in the same output.
    definition['object/bowl']['layers']['16']['features'][0]['b']=2.
    (selection/'concept_features.json').write_text(json.dumps(definition))
    with pytest.raises(ValueError,match='configuration changed'):
        map_concept_ffn.main()
