#!/usr/bin/env python3
import argparse, base64, json, os, pathlib

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--endpoint-id', required=True); ap.add_argument('--image', required=True); ap.add_argument('--prompt', required=True)
    ap.add_argument('--steps', type=int, default=8); ap.add_argument('--duration', type=int, default=15); ap.add_argument('--width', type=int, default=1344); ap.add_argument('--height', type=int, default=768); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--wait', action='store_true'); a=ap.parse_args()
    import requests
    data=base64.b64encode(pathlib.Path(a.image).read_bytes()).decode()
    payload={'input':{'images':[{'name':pathlib.Path(a.image).name,'data':data}], 'params':{'prompt':a.prompt,'steps':a.steps,'duration':a.duration,'width':a.width,'height':a.height,'seed':a.seed}}}
    h={'Authorization':f"Bearer {os.environ['RUNPOD_API_KEY']}",'Content-Type':'application/json'}
    u=f"https://api.runpod.ai/v2/{a.endpoint_id}/run"; r=requests.post(u,headers=h,json=payload,timeout=120); r.raise_for_status(); job=r.json(); print(json.dumps(job,indent=2))
    if a.wait:
        x=requests.get(f"https://api.runpod.ai/v2/{a.endpoint_id}/status/{job['id']}",headers=h,timeout=1800); x.raise_for_status(); print(json.dumps(x.json(),indent=2))
if __name__=='__main__': main()
