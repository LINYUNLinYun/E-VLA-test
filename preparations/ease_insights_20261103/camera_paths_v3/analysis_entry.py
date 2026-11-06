"""Camera discovery report with explicit80-original-Full reference provenance."""
import argparse
import json
from pathlib import Path
from .common import atomic_json,sha
from .selection import analyze_discovery,freeze


def run(root,output,freeze_selection=False):
    report=analyze_discovery(root)
    # The immutable shared selection module retained an old prose label from
    # its template. Counts/selection math already use240+80. State the exact
    # reference population in the separately pinned report entry point.
    report['reference_scope']='80 original Full records retain source identities; no spread_c quality records are referenced and none of these80 controls is a new replication.'
    report['report_entry_source']=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))
    output=Path(output).resolve();atomic_json(output/'analysis.json',report)
    selected=freeze(output,report) if freeze_selection else None
    return dict(complete=report['complete'],new_valid_episodes=report['new_discovery_episodes'],preserved_reference_episodes=report['preserved_reference_episodes'],
        selection_status=report['selection']['status'],selection=str(selected) if selected else None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--freeze-selection',action='store_true');a=parser.parse_args();print(json.dumps(run(a.root,a.output,a.freeze_selection)))
