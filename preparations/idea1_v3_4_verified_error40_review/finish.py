"""Analysis-only repair: parent audit request rows already carry identity fields."""
import hashlib,inspect,time
from pathlib import Path
from preparations.idea1_v3_4_verified_error40 import finish as frozen
from preparations.idea1_v3_4_verified_error40.common import OUT,sha
from preparations.idea1_v3.io import atomic


def main():
    source=inspect.getsource(frozen.main)
    old="requests.extend(dict(condition=cfg['id'],**c,**q) for q in qs)"
    new="requests.extend(qs)"
    assert source.count(old)==1
    repaired=source.replace(old,new)
    receipt=dict(analysis_only=True,source=str(Path(__file__).resolve()),source_sha256=sha(__file__),original_finish_sha256=sha(frozen.__file__),original_manifest_sha256=sha(OUT/'manifest.json'),original_failure_sha256=sha(OUT/'failed.json'),change=dict(before=old,after=new),function_before_sha256=hashlib.sha256(source.encode()).hexdigest(),function_after_sha256=hashlib.sha256(repaired.encode()).hexdigest(),rollout_reruns=0,time=time.time())
    atomic(OUT/'analysis_review.json',receipt)
    def reviewed_atomic(path,data):
        path=Path(path)
        if path.parent==OUT and path.name in ('audit.json','complete.json','status.json'):
            data=dict(data,analysis_review=True,analysis_review_source_sha256=receipt['source_sha256'])
        return atomic(path,data)
    namespace=dict(vars(frozen));namespace['atomic']=reviewed_atomic
    exec(compile(repaired,__file__,'exec'),namespace)
    namespace['main']()
    atomic(OUT/'failure_resolution.json',dict(resolved=True,analysis_only=True,rollout_reruns=0,analysis_review_sha256=sha(OUT/'analysis_review.json'),time=time.time()))

if __name__=='__main__':main()
