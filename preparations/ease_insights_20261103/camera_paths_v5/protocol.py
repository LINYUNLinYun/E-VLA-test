"""Prospective full-fresh-vision plus fixed AE-prefix family."""
import json
import time
from .common import *
REPORT=ROOT/'reports/ease_insights_20261103/camera_paths_v5'


def freeze():
    path=REPORT/'protocol.json'
    if path.exists():return dict(path=str(path),sha256=sha(path))
    prior=ROOT/'reports/ease_insights_20261103/camera_paths_v4/completed_summary.json';old=json.loads(prior.read_text())
    require(old['status']=='complete_no_candidate' and old['new_quality_cases']==160,'Preserve completed fresh-vision negative first')
    historical=ROOT/'reports/ease_insights_20261103/historical_policy_initials_v1/audit.json'
    value=dict(revision='ease-full-fresh-vision-AE-prefix-v5',status='prospective_before_real_GPU_or_new_outcomes',time=time.time(),conditions=conditions(),
        antecedent_evidence=dict(path=str(prior),sha256=sha(prior)),historical_inventory=dict(path=str(historical),sha256=sha(historical)),
        hypothesis='Withfullfreshvision27+27andVLM18prefill,twoequal-sizeAEattentionprefixesmayhavedifferenttask-categorypreferences.NoVLMprefillsavingisclaimed.',
        fixed_science=dict(H=50,R=5,vision=[27,27],VLM_layers=18,AE_layers=18,Full_D=10,Fast_D=2,period=2,
            prefix_patterns=dict(PATTERNS),Fast_visual_keys_per_camera=128,all_text_retained=True,original_decoder_absolute_RoPE=True,training=False,camera_feature_cache=False),
        numerical=dict(native_and_identity=dict(atol=1e-5,rtol=1e-5),
            partial_independent_full_storage_masked_reference=dict(atol=.0009765625,rtol=.0078125,decoded_gripper_all50_exact=True,
                rationale='ProspectiveBF16referencebound1/1024absoluteand1/128relative.Thecompactandfull-storagesoftmax/Vreductionordersdiffer;thisisnotabitwise-equivalenceclaim.Nativeandidentitycontrolsretain1e-5andall8knowncontroltrajectoriesmustbewiseidentical.'),
            structural='ExactoriginalK/Vgathers;actualcompactkeylengths,18AEvisitsperstep,original51absoluteRoPEpositions,fullnativevisionbothfresh,Full+2Fastcadence,bufferandhelperrestoration.'),
        calibration=dict(episodes=8,independent_quality_cases=0,conditions=['full_d10','dense_d2'],tasks=[0],inits=[45],
            bridge='All8raw/decodedknownR5controltrajectoriesbitwiseidenticaltooriginalFull/DenseD2.',
            joint_cost='All4definitionsineachof4completedservices;4realbankinputs,6warmup+12measuredcallsperdefinition/input. Median same-service Full/path>=1.15andpairmax/min<=1.10.Packing+restoreanddecode/transferincluded.DenseD2reported.'),
        discovery=dict(inits=[45,46],all_tasks=40,new_conditions=list(NEW_CONDITIONS),new_rollouts=160,preserved_Full_and_DenseD2=160,analytical_cells=320,
            quality='Eachcandidateoverall>=.75and>=Full-.15',selection='Fixedpaironly;eachoppositesuitegap>=.10,atleast4positivetasksperdirection,everytaskLOOpositive.Rankminimumgap,sumgaps,thenlexicalsuites.Freezeonlyafterall320cellsandindependentaudit.',no_discovery_significance_claim=True),
        confirmation=dict(inits=[10,11,12,13,14],noise_repeat=0,all_tasks=40,conditions=list(CONDITIONS),new_rollouts=800,
            primary='Twodiscovery-frozendirections.One-sidedexactten-tasksignflipincludingties;Holm2.Fivephysicalinitialsremaininsidetheirtask.',
            bootstrap='20000whole-taskinteger-sumresamples/50;seed20261119;linearpercentile.',
            success='Bothdirectionspositive,Holm2p<.05,bothwhole-taskCIlower>0,alltaskLOOpositive;bothcandidateoverall>=.75and>=Full-.15;nativeFull/path>=1.15andpairwithin10%.',
            no_reselection=True,no_added_rollouts_after_significance_inspection=True),
        state_exposure_scope='Allofficial0-49stateshavehistoricalR12exposure.Holdout10-14isseparatefromthisfamily45-46discovery;notgloballynewstates,unseentasksortraining-OOD.No10-14prefix-pathoutcomesmayselectpatternsordirections.',
        physical_contract=dict(environment_seed=7,settling_actions=10,policy_action_limits=LIMITS),maximum_physical_rollouts_including_repeated_controls=968,
        DenseD2_boundary='ReportstrongfixedDenseD2qualityandnativecost.Ifitsuccess/costdominatesboth,donotclaimroutingbenefit.Task-categorypreferenceandbenefitoverbestfixedroutingaredifferentquestions.',
        stop='Rejectfailednumeric/control/costgatesbeforequality.Noqualifieddiscoverypairmeansnoconfirmation.Failedconfirmationremainsnot_reproduced.Noeffectivefailurererunsorthresholdrelaxation.',
        overall_goal_complete_claim=False,limits='Separatelyversionedexplorationthenfixed-pairreplication;nostudy-wide5percenterrorclaim,onlinephaseselectororexclusiveGPU/closedloopspeedclaim.')
    atomic_json(path,value);return dict(path=str(path),sha256=sha(path))
