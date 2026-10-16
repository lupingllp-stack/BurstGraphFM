import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
import yaml
from .data import GraphDataset, collate, split_groups, fit_normalizer
from .model import BurstClassifier, StructuralBurstClassifier

def setup(seed):
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'))
    if world>1:
        if not torch.cuda.is_available():raise RuntimeError('DDP currently requires CUDA')
        local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
        dist.init_process_group('nccl');device=torch.device('cuda',local)
    else:device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    return rank,world,device

def scores(logits,labels,classes):
    pred=logits.argmax(-1);acc=(pred==labels).float().mean().item()
    f1=[]
    for c in range(classes):
        tp=((pred==c)&(labels==c)).sum().item()
        fp=((pred==c)&(labels!=c)).sum().item()
        fn=((pred!=c)&(labels==c)).sum().item()
        f1.append(2*tp/max(2*tp+fp+fn,1))
    return {'accuracy':acc,'macro_f1':sum(f1)/classes,'class_f1':f1}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/baseline.yaml')
    p.add_argument('--model',choices=['sequence','graph','structural'],default='graph')
    p.add_argument('--output',default='runs/graph')
    p.add_argument('--pretrained',default=None,help='Pretraining best.pt; loads encoder and its normalizer')
    a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text())
    rank,world,device=setup(cfg['seed'])
    try:
        blob=torch.load(cfg['dataset'],map_location='cpu',weights_only=False)
        graphs=blob['graphs'];classes=len(blob['labels'])
        splits,group_names=split_groups(graphs,cfg['seed'])
        mu,sigma=fit_normalizer(splits[0])
        pre=None
        if a.pretrained:
            pre=torch.load(a.pretrained,map_location='cpu',weights_only=False)
            pc=pre['config']
            for key in ('hidden','layers','heads'):
                if cfg[key]!=pc[key]:raise ValueError(f'Model {key} must equal pretraining value {pc[key]}')
            mu,sigma=pre['mu'],pre['sigma']
        train,val,test=[GraphDataset(s,mu,sigma) for s in splits]
        tr_sampler=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=cfg['seed']) if world>1 else None
        tr=DataLoader(train,batch_size=cfg['batch_size'],shuffle=tr_sampler is None,sampler=tr_sampler,collate_fn=collate,num_workers=cfg.get('workers',0))
        va=DataLoader(val,batch_size=cfg['batch_size'],collate_fn=collate)
        te=DataLoader(test,batch_size=cfg['batch_size'],collate_fn=collate)
        if a.model=='structural':
            top_k=(pre['config'].get('interaction_top_k',3) if pre else cfg.get('interaction_top_k',3))
            net=StructuralBurstClassifier(classes,cfg['hidden'],cfg['layers'],cfg['heads'],cfg['dropout'],top_k).to(device)
        else:
            net=BurstClassifier(classes,a.model,cfg['hidden'],cfg['layers'],cfg['heads'],cfg['dropout']).to(device)
        if pre is not None:
            missing,unexpected=net.load_state_dict(pre['encoder'],strict=False)
            assert all(k.startswith('head.') or (a.model!='structural' and k.startswith('interaction.')) for k in missing) and not unexpected,(missing,unexpected)
            if a.model=='structural' and not pre['config'].get('structural',False):
                raise ValueError('Structural fine-tuning requires structural pretraining weights')
            if rank==0:print('Loaded pretrained encoder:',a.pretrained,flush=True)
        model=DistributedDataParallel(net,device_ids=[device.index]) if world>1 else net
        opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['lr']),weight_decay=float(cfg['weight_decay']))
        use_amp=device.type=='cuda';best=-1.;out=Path(a.output)
        if rank==0:
            out.mkdir(parents=True,exist_ok=True)
            (out/'split.json').write_text(json.dumps({'groups':group_names,'labels':blob['labels'],'counts':[len(s) for s in splits]},indent=2))
        def evaluate(loader):
            net.eval();pred=[];truth=[];loss_sum=0.;n=0
            with torch.no_grad():
                for batch in loader:
                    batch={k:v.to(device) for k,v in batch.items()}
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=use_amp):
                        y=net(batch['x'],batch['edge'],batch['mask'])
                        loss=torch.nn.functional.cross_entropy(y,batch['y'])
                    pred.append(y.float().cpu());truth.append(batch['y'].cpu())
                    loss_sum+=loss.item()*len(batch['y']);n+=len(batch['y'])
            return {'loss':loss_sum/max(n,1),**scores(torch.cat(pred),torch.cat(truth),classes)}
        patience=0
        for epoch in range(cfg['epochs']):
            model.train()
            if tr_sampler:tr_sampler.set_epoch(epoch)
            total=0.;count=0
            for batch in tr:
                batch={k:v.to(device) for k,v in batch.items()}
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=use_amp):
                    y=model(batch['x'],batch['edge'],batch['mask'])
                    loss=torch.nn.functional.cross_entropy(y,batch['y'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
                opt.step()
                total+=loss.item()*len(batch['y']);count+=len(batch['y'])
            stop=torch.zeros(1,device=device)
            if rank==0:
                vm=evaluate(va)
                print(json.dumps({'epoch':epoch+1,'train_loss_rank0':round(total/max(count,1),5),'val':vm},ensure_ascii=False),flush=True)
                if vm['macro_f1']>best:
                    best=vm['macro_f1'];patience=0
                    torch.save({'model':net.state_dict(),'config':cfg,'model_type':a.model,'mu':mu,'sigma':sigma,
                                'labels':blob['labels'],'best_val':vm},out/'best.pt')
                else:patience+=1
                if patience>=cfg['patience']:stop.fill_(1)
            if world>1:dist.broadcast(stop,src=0)
            if stop.item():break
        if world>1:dist.barrier()
        if rank==0:
            checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=False)
            net.load_state_dict(checkpoint['model'])
            result={'val':evaluate(va),'test':evaluate(te),'split':group_names,'model':a.model}
            (out/'metrics.json').write_text(json.dumps(result,indent=2))
            print('FINAL',json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
