"""Visual triage from the five inspected contact sheets; not defect annotation."""
from pathlib import Path
import csv, json
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'data/custom_unannotated_v1/manifest.csv'
with p.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
non_target={38,39,40,41,42,43,57,58,59,90,186,213}
for i,r in enumerate(rows,1):
    group='indoor_pink_airframe' if i<=37 else 'indoor_blue_white_airframe' if 44<=i<=56 else 'outdoor_transport_airframe' if i>=60 else 'non_target_context'
    r['contact_index']=i
    r['visual_group']=group
    r['group']='unresolved_indoor_session' if i<60 else 'outdoor_transport_session'
    r['aircraft_id']='needs_user_mapping'
    r['curation_status']='exclude_non_target_candidate' if i in non_target else 'retain_for_annotation_review'
    r['group_provenance']='assistant_visual_contact_sheet_review_not_confirmed_aircraft_identity'
with p.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
summary={'total':len(rows),'retained_candidates':sum(r['curation_status']=='retain_for_annotation_review' for r in rows),'non_target_candidates':sorted(non_target),'user_context':'Two aircraft: glider and retired transport; synchronized IMU/VIO to be supplied later. Exact file mapping not supplied.','limitations':'Visual groups are provisional. Every retained image still requires defect annotation; no negative labels inferred.'}
(ROOT/'reports/custom_curation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
