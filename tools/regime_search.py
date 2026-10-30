"""
Fast search for GROKKING configs: train with no checkpointing, report whether and
when test accuracy takes off. Cheap enough to scan a grid before committing CPU to
full trajectory-saving runs.

A run 'groks' if train acc hits ~1.0 early and test acc reaches >=0.9 much later.
"""
import sys, os, json, argparse, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0,"."); sys.path.insert(0,"tools")
import numpy as np, torch
from train_topology import build_task
from models import MLP
from utils import stablemax_cross_entropy, softmax_cross_entropy

def run(task,p,sym_n,tf,wd,hid,lr,epochs,seed,loss="stablemax"):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(2)
    pairs,y,n,_=build_task(task,p,sym_n)
    N=len(y); X=torch.zeros(N,2*n)
    X[torch.arange(N),torch.as_tensor(pairs[:,0])]=1.; X[torch.arange(N),n+torch.as_tensor(pairs[:,1])]=1.
    Y=torch.as_tensor(y,dtype=torch.long)
    g=torch.Generator().manual_seed(seed); perm=torch.randperm(N,generator=g)
    ntr=int(tf*N); tr,te=perm[:ntr],perm[ntr:]
    m=MLP(input_size=2*n,hidden_sizes=[hid,hid],output_size=n,bias=False)
    o=torch.optim.AdamW(m.parameters(),lr=lr,betas=(0.9,0.999),eps=1e-25,weight_decay=wd)
    L=stablemax_cross_entropy if loss=="stablemax" else softmax_cross_entropy
    ep_tr1=None; ep_te90=None; best=0.
    for ep in range(epochs+1):
        m.train(); o.zero_grad(); l=L(m(X[tr]),Y[tr],dtype=torch.float64); l.backward(); o.step()
        if ep%250==0:
            m.eval()
            with torch.no_grad():
                tra=(m(X[tr]).argmax(1)==Y[tr]).float().mean().item()
                tea=(m(X[te]).argmax(1)==Y[te]).float().mean().item()
            best=max(best,tea)
            if ep_tr1 is None and tra>=0.99: ep_tr1=ep
            if ep_te90 is None and tea>=0.90: ep_te90=ep; break
    return {"task":task,"p":n,"tf":tf,"wd":wd,"hidden":hid,"lr":lr,"seed":seed,
            "ep_train1":ep_tr1,"ep_test90":ep_te90,"best_test":round(best,3),
            "delay":(ep_te90-ep_tr1) if (ep_te90 and ep_tr1) else None}

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--shard",type=int,default=0); ap.add_argument("--nshard",type=int,default=1)
    a=ap.parse_args()
    cfgs=[]
    for task,p,sym in (("add",23,5),("perm",0,4)):
        for tf in (0.6,0.7,0.8):
            for wd in (0.01,0.03):
                for hid in (48,200):
                    cfgs.append(dict(task=task,p=p,sym_n=sym,tf=tf,wd=wd,hid=hid,lr=0.01,epochs=20000,seed=0))
    out=[]
    for i,c in enumerate(cfgs):
        if i%a.nshard!=a.shard: continue
        t0=time.time(); r=run(**c); r["secs"]=round(time.time()-t0)
        print(json.dumps(r),flush=True); out.append(r)
