"""Label-free Masked Burst Modeling + masked temporal edge reconstruction."""
import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from .data import GraphDataset, collate, fit_normalizer, split_groups
from .model import Block

CONTINUOUS = [0,1,2,3,4,5,6,7,8,9,10,13,14,15]

class PretrainNet(nn.Module):
    def __init__(self, hidden=128, layers=4, heads=4, dropout=0.1, model_type='graph'):
        super().__init__()
        self.model_type=model_type
        self.input=nn.Linear(16,hidden)
        self.mask_token=nn.Parameter(torch.zeros(hidden))
        self.blocks=nn.ModuleList([Block(hidden,heads,dropout) for _ in range(layers)])
        self.node_head=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,16))
        self.edge_head=nn.Sequential(nn.LayerNorm(hidden*2),nn.Linear(hidden*2,hidden),nn.GELU(),nn.Linear(hidden,1))

    def forward(self,x,edge,valid,masked_nodes,pairs):
        h=self.input(x)
        h=torch.where(masked_nodes.unsqueeze(-1),self.mask_token.view(1,1,-1),h)
        for block in self.blocks:h=block(h,edge,valid,self.model_type=='graph')
        node_output=self.node_head(h)
        b,i,j=pairs
        pair_output=self.edge_head(torch.cat((h[b,i],h[b,j]),dim=-1))
        return node_output,pair_output

def create_masks(x,edge,valid,node_ratio,edge_ratio):
    B,N,_=x.shape
    node_mask=torch.zeros_like(valid)
    masked_edge=edge.clone()
    triples=[]
    for b in range(B):
        n=int(valid[b].sum())
        if n==0:continue
        node_count=max(1,round(n*node_ratio))
        ids=torch.randperm(n,device=x.device)[:node_count]
        node_mask[b,ids]=True
        if n>1:
            edge_count=max(1,round((n-1)*edge_ratio))
            eids=torch.randperm(n-1,device=x.device)[:edge_count]
            for idx in eids.tolist():
                triples.append((b,idx,idx+1))
                masked_edge[b,idx,idx+1]=0
                masked_edge[b,idx+1,idx]=0
    if triples:
        pairs=tuple(torch.tensor([t[i] for t in triples],device=x.device,dtype=torch.long) for i in range(3))
    else:
        empty=torch.zeros(0,dtype=torch.long,device=x.device)
        pairs=(empty,empty,empty)
    return node_mask,masked_edge,pairs

def objective(net,batch,node_ratio=0.2,edge_ratio=0.2,edge_weight=1.):
    x,edge,valid=batch['x'],batch['edge'],batch['mask']
    nm,masked_edge,pairs=create_masks(x,edge,valid,node_ratio,edge_ratio)
    x_visible=x.clone()
    if pairs[0].numel():
        b,i,j=pairs
        x_visible[b,j,14]=0.
    pred,pred_edge=net(x_visible,masked_edge,valid,nm,pairs)
    cont=F.smooth_l1_loss(pred[nm][:,CONTINUOUS],x[nm][:,CONTINUOUS])
    target=x[nm][:,11:13].argmax(dim=-1)
    direction=F.cross_entropy(pred[nm,11:13],target)
    if pairs[0].numel():
        b,i,j=pairs
        relation=F.smooth_l1_loss(pred_edge[:,0],edge[b,i,j,1])
    else:relation=pred_edge.sum()*0.
    loss=cont+direction+edge_weight*relation
    return loss,{'node_reg':float(cont.detach()),'direction':float(direction.detach()),'temporal_relation':float(relation.detach())}

def load_data(path,seed,holdout_groups=None):
    blob=torch.load(path,map_location='cpu',weights_only=False)
    graphs=blob['graphs']
    splits,names=split_groups(graphs,seed)
    if holdout_groups:
        excluded=set(holdout_groups)
        keep=[g for g in splits[0] if g['group'] not in excluded]
        if not keep:raise ValueError('No pretraining graphs after excluding groups')
        splits[0]=keep
    mu,sigma=fit_normalizer(splits[0])
    return splits,names,mu,sigma

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/pretrain.yaml')
    ap.add_argument('--output',default='runs/pretrain')
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    rank=int(os.environ.get('RANK',0))
    world=int(os.environ.get('WORLD_SIZE',1))
    local=int(os.environ.get('LOCAL_RANK',0))
    if world>1:
        if not torch.cuda.is_available():raise RuntimeError('DDP needs CUDA')
        torch.cuda.set_device(local)
        dist.init_process_group('nccl')
    device=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    seed=cfg['seed']
    random.seed(seed+rank)
    np.random.seed(seed+rank)
    torch.manual_seed(seed+rank)
    try:
        splits,group_names,mu,sigma=load_data(cfg['dataset'],seed)
        train=GraphDataset(splits[0],mu,sigma)
        val=GraphDataset(splits[1],mu,sigma)
        sampler=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=seed) if world>1 else None
        tr=DataLoader(train,batch_size=cfg['batch_size'],sampler=sampler,shuffle=sampler is None,collate_fn=collate,num_workers=cfg.get('workers',0))
        va=DataLoader(val,batch_size=cfg['batch_size'],shuffle=False,collate_fn=collate)
        if cfg.get('structural',False):
            from .structural import StructuralPretrainNet
            net=StructuralPretrainNet(cfg['hidden'],cfg['layers'],cfg['heads'],cfg['dropout'],
                top_k=cfg.get('interaction_top_k',3),temperature=cfg.get('temperature',0.2)).to(device)
        else:
            net=PretrainNet(cfg['hidden'],cfg['layers'],cfg['heads'],cfg['dropout'],cfg.get('model_type','graph')).to(device)
        model=DDP(net,device_ids=[local],find_unused_parameters=False) if world>1 else net
        opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['lr']),weight_decay=float(cfg['weight_decay']))
        out=Path(args.output)
        if rank==0:
            out.mkdir(parents=True,exist_ok=True)
            (out/'split.json').write_text(json.dumps({'groups':group_names,'counts':list(map(len,splits))},indent=2))
        best=float('inf')
        patience=0
        def run_epoch(loader,training):
            model.train(training)
            total=0.
            count=0
            for batch in loader:
                batch={k:v.to(device) for k,v in batch.items()}
                if training:opt.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
                        if cfg.get('structural',False):
                            from .structural import structural_objective
                            loss,_=structural_objective(model if training else net,batch,
                                cfg['mask_ratio'],cfg['edge_mask_ratio'],cfg['edge_weight'],
                                cfg.get('contrast_weight',0.5),cfg.get('consistency_weight',0.1))
                        else:
                            loss,_=objective(model if training else net,batch,cfg['mask_ratio'],cfg['edge_mask_ratio'],cfg['edge_weight'])
                    if training:
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
                        opt.step()
                total+=float(loss.detach())*len(batch['x'])
                count+=len(batch['x'])
            return total/max(count,1)
        for ep in range(cfg['epochs']):
            if sampler:sampler.set_epoch(ep)
            train_loss=run_epoch(tr,True)
            stop=torch.zeros(1,device=device)
            if rank==0:
                with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
                    torch.manual_seed(12345)
                    validation=run_epoch(va,False)
                print(json.dumps({'epoch':ep+1,'train_rank0':train_loss,'val':validation}),flush=True)
                if validation<best:
                    best=validation
                    patience=0
                    encoder={k:v.cpu() for k,v in net.state_dict().items() if k.startswith(('input.','blocks.','interaction.'))}
                    torch.save({'encoder':encoder,'model':net.state_dict(),'mu':mu,'sigma':sigma,
                                'config':cfg,'groups':group_names,'best_val':best},out/'best.pt')
                else:patience+=1
                if patience>=cfg['patience']:stop.fill_(1.)
            if world>1:dist.broadcast(stop,src=0)
            if stop.item():break
        if rank==0:print('PRETRAIN COMPLETE:',out/'best.pt',flush=True)
    finally:
        if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
