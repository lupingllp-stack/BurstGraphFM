"""Deterministic bidirectional flow -> direction-change Burst -> graph tensors."""
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from .pcap import iter_packets

FEATURE_NAMES = ['log_packets', 'log_bytes', 'mean_len', 'std_len', 'min_len', 'max_len',
                 'log_duration', 'mean_iat', 'std_iat', 'min_iat', 'max_iat',
                 'c2s', 's2c', 'relative_index', 'log_prev_gap', 'byte_fraction']

def flow_key(src, dst, proto):
    a, b = sorted((src, dst))
    return (proto, a, b)

def make_graph(packets, max_bursts=64):
    packets = sorted(packets, key=lambda p: p[0])
    if not packets:return None
    bursts = []
    for p in packets:
        if not bursts or p[1] != bursts[-1][0][1]:bursts.append([p])
        else:bursts[-1].append(p)
    original_n=len(bursts)
    bursts=bursts[:max_bursts]
    total_bytes=max(sum(p[2] for b in bursts for p in b),1)
    features=[]
    for i,b in enumerate(bursts):
        times=np.asarray([p[0] for p in b],dtype=np.float64)
        lengths=np.asarray([p[2] for p in b],dtype=np.float64)
        iat=np.diff(times)
        gap=max(0.,times[0]-bursts[i-1][-1][0]) if i else 0.
        duration=max(0.,times[-1]-times[0])
        values=[np.log1p(len(b)),np.log1p(lengths.sum()),lengths.mean(),lengths.std(),
                lengths.min(),lengths.max(),np.log1p(duration),
                np.log1p(iat.mean()) if len(iat) else 0.,
                np.log1p(iat.std()) if len(iat) else 0.,
                np.log1p(iat.min()) if len(iat) else 0.,
                np.log1p(iat.max()) if len(iat) else 0.,
                float(b[0][1]==0),float(b[0][1]==1),
                i/max(len(bursts)-1,1),np.log1p(gap),lengths.sum()/total_bytes]
        features.append(values)
    n=len(bursts)
    edges=np.zeros((n,n,4),dtype=np.float32)
    for i in range(n-1):
        gap=max(0.,bursts[i+1][0][0]-bursts[i][-1][0])
        for a,b in ((i,i+1),(i+1,i)):
            edges[a,b]=[1.,np.log1p(gap),float(bursts[i][0][1]!=bursts[i+1][0][1]),float(b-a)]
    return dict(x=torch.tensor(np.asarray(features),dtype=torch.float32),
                edge=torch.tensor(edges),n_original=original_n)

def process_capture(path,label,max_bursts=64,flow_timeout=120.,min_packets=3,max_packets=10000,max_flows=0):
    active={};completed=[]
    for ts,src,dst,proto,size in iter_packets(path):
        key=flow_key(src,dst,proto)
        state=active.get(key)
        if state is not None and (ts-state['last']>flow_timeout or ts<state['last']):
            if len(state['packets'])>=min_packets:completed.append(state['packets'])
            state=None
        if state is None:
            state={'origin':src,'last':ts,'packets':[]}
            active[key]=state
        direction=0 if src==state['origin'] else 1
        if len(state['packets'])<max_packets:state['packets'].append((ts,direction,size))
        state['last']=ts
    completed += [v['packets'] for v in active.values() if len(v['packets'])>=min_packets]
    if max_flows:completed=completed[:max_flows]
    graphs=[]
    for idx,pkt in enumerate(completed):
        g=make_graph(pkt,max_bursts)
        if g is not None:
            g.update(y=int(label),capture=str(Path(path).name),group=Path(path).stem,flow_index=idx)
            graphs.append(g)
    return graphs

def build_manifest(manifest,output,max_bursts=64,flow_timeout=120.,min_packets=3,max_flows_per_capture=0):
    rows=json.loads(Path(manifest).read_text())
    labels=sorted(set(str(r['label']) for r in rows))
    label_map={name:i for i,name in enumerate(labels)}
    all_graphs=[]
    for r in rows:
        path=Path(r['path'])
        graphs=process_capture(path,label_map[str(r['label'])],max_bursts,flow_timeout,min_packets,
                               max_flows=max_flows_per_capture)
        group=str(r.get('group',path.stem))
        for g in graphs:g['group']=group
        print(f'{path.name}: {len(graphs)} flows, group={group}',flush=True)
        all_graphs+=graphs
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'graphs':all_graphs,'labels':labels,'features':FEATURE_NAMES,
                'manifest':rows,'schema_version':1},out)
    print('Saved',out,'graphs=',len(all_graphs),'labels=',labels)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',default='data/processed/graphs.pt')
    p.add_argument('--max-bursts',type=int,default=64)
    p.add_argument('--flow-timeout',type=float,default=120.)
    p.add_argument('--min-packets',type=int,default=3)
    p.add_argument('--max-flows-per-capture',type=int,default=0)
    a=p.parse_args()
    build_manifest(a.manifest,a.output,a.max_bursts,a.flow_timeout,a.min_packets,a.max_flows_per_capture)
if __name__=='__main__':main()
