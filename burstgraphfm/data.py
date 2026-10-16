import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self,graphs,mu=None,sigma=None):
        self.graphs=graphs;self.mu=mu;self.sigma=sigma
    def __len__(self):return len(self.graphs)
    def __getitem__(self,i):
        g=self.graphs[i];x=g['x'].clone()
        if self.mu is not None:x=(x-self.mu)/self.sigma
        return {'x':x,'edge':g['edge'],'y':g['y']}

def collate(rows):
    b=len(rows);n=max(r['x'].shape[0] for r in rows);f=rows[0]['x'].shape[-1]
    x=torch.zeros(b,n,f);e=torch.zeros(b,n,n,4);mask=torch.zeros(b,n,dtype=torch.bool)
    y=torch.zeros(b,dtype=torch.long)
    for i,r in enumerate(rows):
        z=len(r['x']);x[i,:z]=r['x'];e[i,:z,:z]=r['edge'];mask[i,:z]=True;y[i]=r['y']
    return {'x':x,'edge':e,'mask':mask,'y':y}

def split_groups(graphs,seed=42,ratios=(0.7,0.15,0.15)):
    """Group-disjoint, class-stratified capture split for pure-label capture groups."""
    from collections import Counter,defaultdict
    by_group=defaultdict(list)
    for g in graphs:by_group[g['group']].append(g)
    by_class=defaultdict(list)
    for group,items in by_group.items():
        labels=set(g['y'] for g in items)
        if len(labels)!=1:
            raise ValueError(f'Mixed-label capture group {group}: manual group split needed')
        by_class[next(iter(labels))].append(group)
    rng=random.Random(seed)
    names=[set(),set(),set()]
    for label,groups in sorted(by_class.items()):
        if len(groups)<3:
            raise ValueError(f'Class {label} has only {len(groups)} groups; need >=3 independent capture groups per class')
        groups=sorted(groups);rng.shuffle(groups)
        n=len(groups);ntr=max(1,round(n*ratios[0]));nval=max(1,round(n*ratios[1]))
        if ntr+nval>=n:ntr=n-2;nval=1
        for dest,part in zip(names,(groups[:ntr],groups[ntr:ntr+nval],groups[ntr+nval:])):
            dest.update(part)
    partitions=[[g for g in graphs if g['group'] in s] for s in names]
    return partitions,[sorted(s) for s in names]

def fit_normalizer(graphs):
    x=torch.cat([g['x'] for g in graphs])
    mu=x.mean(0);sigma=x.std(0,unbiased=False).clamp_min(1e-6)
    mu[11:13]=0;sigma[11:13]=1
    return mu,sigma
