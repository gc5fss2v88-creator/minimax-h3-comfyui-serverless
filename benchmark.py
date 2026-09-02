#!/usr/bin/env python3
"""Run the requested matrix against a deployed endpoint; saves raw JSON for comparison."""
import argparse, json, os, pathlib, subprocess, time
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--endpoint-id',required=True); ap.add_argument('--image',required=True); ap.add_argument('--prompt',required=True); ap.add_argument('--attention',type=int,choices=(0,1),required=True,help='Deploy the endpoint with ENABLE_ATTENTION set to this value before running'); ap.add_argument('--out',default='benchmark-results.json'); a=ap.parse_args()
    rows=[]
    for steps in (20,8,6,4):
        t=time.time(); cmd=['python3','scripts/submit_runpod.py','--endpoint-id',a.endpoint_id,'--image',a.image,'--prompt',a.prompt,'--steps',str(steps),'--wait']; env={**os.environ}
        p=subprocess.run(cmd,text=True,capture_output=True,env=env); rows.append({'attention':a.attention,'steps':steps,'elapsed_submit_wait_s':round(time.time()-t,2),'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
    pathlib.Path(a.out).write_text(json.dumps(rows,ensure_ascii=False,indent=2)); print(a.out)
if __name__=='__main__': main()
