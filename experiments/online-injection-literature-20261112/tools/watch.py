from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','formal','diagnostic']);p.add_argument('--via-jump',action='store_true');a=p.parse_args()
base=Path(__file__).resolve().parent
failures=0;last=None
deadline=time.monotonic()+24*3600
while time.monotonic()<deadline:
    r=subprocess.run([sys.executable,'-X','utf8',str(base.parent/'online_injection_scaling_20261109/remote.py'),
        'a',*(['--via-jump'] if a.via_jump else []),'--file',str(base/'monitor_remote.py'),'--timeout','45'],capture_output=True,timeout=55)
    if r.returncode:
        failures+=1
        print(json.dumps({'transport_failures':failures,'stderr':r.stderr.decode(errors='replace')[-300:]}),flush=True)
        if failures>=3:raise SystemExit(2)
    else:
        failures=0
        rows=[x for x in map(json.loads,r.stdout.splitlines()) if x.get('stage')==a.stage]
        compact=[{k:x[k] for k in ('worker','counts','last','empty_candidates','truncated_candidates','exit')} for x in rows]
        value=json.dumps(compact)
        if value!=last:
            print(value,flush=True);last=value
        if any(x['counts'].get('failed') or x['exit'] not in (None,'0') for x in rows):
            print(r.stdout.decode(),flush=True);raise SystemExit(1)
        if len(rows)==1 and all(x['exit']=='0' for x in rows):raise SystemExit(0)
    time.sleep(30)
raise SystemExit('watch deadline reached; model jobs remain untouched')
