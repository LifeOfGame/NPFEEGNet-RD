from pathlib import Path
from datetime import datetime,timezone
import json
R=Path(__file__).resolve().parent
state=json.loads((R/'status.json').read_text());elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(state.get('started_at',state['updated_at']))).total_seconds()
maps={v:json.loads((R/f'{v}_epochs.json').read_text())['selected_epochs'] for v in ['rd','no']}
totals={v:sum(int(e) for seeds in m.values() for e in seeds.values()) for v,m in maps.items()}
arms={};lane_done={v:0 for v in ['rd','no']}
for arm in ['rd_at_rd','rd_at_no','no_at_rd','no_at_no']:
 variant,duration=arm.split('_at_');paths=list((R/'output'/arm).glob('NPFEEGNet/*/sub*/seed*/test_auxiliary.npz'))
 done=sum(int(maps[duration][p.parent.parent.name[3:]][p.parent.name[4:]]) for p in paths)
 lane_done[variant]+=done;arms[arm]={'completed':len(paths),'epochs_completed':done,'epochs_total':totals[duration]}
eta=[elapsed*(sum(totals.values())/n-1)/3600 for n in lane_done.values() if n]
print(json.dumps({'stage':state['stage'],'elapsed_minutes':round(elapsed/60,1),'arms':arms,'completed':sum(a['completed'] for a in arms.values()),'total':648,'estimated_remaining_hours_by_epoch_work':round(max(eta),2) if len(eta)==2 else None},indent=2))
