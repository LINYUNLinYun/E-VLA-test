"""Freeze the fresh-camera allocation family before any real numerical or quality runs."""
import json
import time
from .common import *
REPORT=ROOT/'reports/ease_insights_20261103/camera_paths_v4'


def freeze():
    target=REPORT/'protocol.json'
    if target.exists():return dict(path=str(target),sha256=sha(target))
    prior=ROOT/'reports/ease_insights_20261103/ae_complement_beam_v4/completed_summary.json';previous=json.loads(prior.read_text())
    require(previous['status']=='complete_not_reproduced' and previous['new_quality_cases']==720,'Finish and preserve prior registered negative first')
    historical=ROOT/'reports/ease_insights_20261103/historical_policy_initials_v1/audit.json';h=json.loads(historical.read_text());require(h['status']=='metadata_inventory_passed','Need prior-use scope disclosure')
    value=dict(revision='ease-fresh-camera-depth-v4',status='prospective_before_real_GPU_or_new_outcomes',time=time.time(),conditions=conditions(),
        antecedent_evidence=dict(path=str(prior),sha256=sha(prior)),historical_inventory=dict(path=str(historical),sha256=sha(historical)),
        hypothesis='At the same45 logical image-layer budget,two fresh-camera depth allocations may have different task-category preferences. This is not a stale-camera cache hypothesis.',
        fixed_science=dict(H=50,R=5,vision_Full=[27,27],vision_Fast_choices=[[27,18],[18,27]],sparse_skip_original_layers=list(SPARSE_SKIP),VLM_layers=18,AE_layers=18,Full_D=10,Fast_D=2,period=2,training=False,compensation=None,fresh_cameras=[0,1],camera_feature_cache=False),
        numerical='OriginalnativeB2andall-kepttwoB1rawvision/prefix/rawactionsparity1e-5;partialB1allocationversusseparateB2compute-all/restore-skipped-vieworacle;actualbatch1tensors,originalvisionweightindices,AE18NFE,nodes,cadenceandhelperrestoration.',
        calibration=dict(episodes=8,independent_quality_cases=0,conditions=['full_d10','dense_d2'],tasks=[0],inits=[45],
            bridge='All8raw/decodedknownR5controltrajectoriesbitwiseidenticaltooriginalFull/DenseD2;emptyVLMmaskmakesoldextrapolationcompensationinert.',
            joint_cost='All4definitionsineachof4completedservices;fourrealbankinputs,6warmups+12measuredcallsperdefinition/input. Median same-service Full/path ratios>=1.15andpairmax/min<=1.10 for the two allocation paths. DenseD2costisalsoexplicitlyreported.'),
        discovery=dict(inits=[45,46],all_tasks=40,new_conditions=list(NEW_CONDITIONS),new_rollouts=160,preserved_Full_and_DenseD2=160,analytical_cells=320,
            quality='Eachcandidateoverall>=.75and>=Full-.15',
            selection='Fixedpaironly;allorderedoppositesuitecontrasts.Requireeachgap>=.10,atleast4positive tasksperdirection,everytaskLOOpositive. Rankminimumgap,sumgaps,thenlexicalsuites.Freezeafterall320cellsandindependentaudit.',
            no_discovery_significance_claim=True),
        confirmation=dict(inits=[10,11,12,13,14],noise_repeat=0,all_tasks=40,conditions=list(CONDITIONS),new_rollouts=800,
            roles='Full,DenseD2strongfixedcontrol,andthetwoallocationpaths;200distinctphysicalcasespercondition.Oneoriginalcondition-independentnoisestream.',
            primary='Twodiscovery-frozendirections.One-sidedexactten-tasksignflipincludingties;Holm2.Fiveinitialstatesstayinsideeachtask.',
            bootstrap='20000whole-taskinteger-sumresamplesdividedby50;seed20261117;linearpercentile.',
            success='Bothdirectionspositive,primaryHolm2p<.05,bothwhole-taskCIlower>0,everytaskLOOpositive;bothcandidateoverall>=.75and>=Full-.15;newnativeFull/path>=1.15andpairwithin10%.',
            no_reselection=True,no_added_rollouts_after_significance_inspection=True),
        state_exposure_scope='Allofficial0-49stateshavehistoricalR12exposure. Thisfamilyholdout10-14isseparatefromits45-46discovery;itdoesnotclaimgloballynovelstates,unseentasks,orabsenceoftrainingexposure. Noallocation-pathoutcomesfromtheholdoutmayselectmasks/directions.',
        physical_contract=dict(environment_seed=7,settling_actions=10,policy_action_limits=LIMITS),
        maximum_physical_rollouts_including_repeated_controls=968,
        DenseD2_boundary='ReportDenseD2qualityandnativecosts.Ifitdominatesbothatnohighercost,donotclaimanadaptive-routingbenefitfromthepairalone.Task-categorypreferenceandbest-fixedroutinggainaredifferentquestions.',
        stop='Rejectnumeric,controlorcostfailurebeforequalitydiscovery.Noqualifieddiscoverypairmeansnoconfirmation;failedconfirmationremainsnot_reproduced.Donotrerunvalidfailuresorrelaxthresholds.',
        overall_goal_complete_claim=False,current_old720_sources_or_verdict_changed=False,
        limits='Thisisaseparatelyversionedexploratoryfamilyfollowedbyfixed-pairreplication.Noonlinephase/router,exclusiveGPUacceleration,universalcategorygeneralization,orstudy-wide5percenterrorclaim.')
    atomic_json(target,value);return dict(path=str(target),sha256=sha(target))
