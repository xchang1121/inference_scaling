from pathlib import Path
import argparse
import os
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','formal']);a=p.parse_args()
base=Path(__file__).resolve().parent
repo=base.parent/'online_injection_20261109/source'
env=dict(os.environ)
env['PYTHONPATH']=os.pathsep.join(map(str,[repo/'src',repo/'scripts',repo.parent/'devdeps']))
command=[sys.executable,'-X','utf8',str(repo/'scripts/summarize_literature_retrieval.py'),
    '--snapshot',str(base/'snapshot'/a.stage),'--output',str(base/('reports-'+a.stage)),
    '--gold',str(base/'data/gold.jsonl'),'--stage',a.stage,'--frozen',str(base/'FROZEN_IS.json'),
    '--manifest',str(base/'data/manifest.json')]
raise SystemExit(subprocess.run(command,env=env).returncode)
