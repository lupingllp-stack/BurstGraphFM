"""Structure-aware SSL with cross-view graph contrast and non-local edge learning."""
import torch
from torch import nn
from torch.nn import functional as F
from .interaction import LearnedInteraction, relation_stability
from .pretrain import PretrainNet, create_masks, CONTINUOUS

class StructuralPretrainNet(PretrainNet):
    def __init__(self,hidden=128,layers=4,heads=4,dropout=0.1,top_k=3,temperature=0.2):
        super().__init__(hidden,layers,heads,dropout,'graph')
        self.interaction=LearnedInteraction(hidden,heads,top_k=top_k)
        self.projection=nn.Sequential(nn.LayerNorm(hidden*2),nn.Linear(hidden*2,hidden),
                                      nn.GELU(),nn.Linear(hidden,hidden))
        self.temperature=temperature

    def encode_view(self,x,edge,valid,masked_nodes):
        h=self.input(x)
        h=torch.where(masked_nodes.unsqueeze(-1),self.mask_token.view(1,1,-1),h)
        bias,selected,score,eligible=self.interaction(h,valid)
        augmented_edge=edge.clone()
        for block in self.blocks:
            h=block(h,augmented_edge,valid,True,extra_bias=bias)
        denom=valid.sum(dim=1,keepdim=True).clamp_min(1)
        mean=(h*valid.unsqueeze(-1)).sum(dim=1)/denom
        maximum=h.masked_fill(~valid.unsqueeze(-1),torch.finfo(h.dtype).min).max(dim=1).values
        pooled=torch.cat((mean,maximum),dim=-1)
        return h,F.normalize(self.projection(pooled),dim=-1),score,selected

def graph_info_nce(z1,z2,temperature=0.2):
    """Each flow is a positive across views. Other flows in the batch are negatives."""
    if z1.shape[0]<2:return (z1.sum()+z2.sum())*0.
    logits=z1.float()@z2.float().T/temperature
    target=torch.arange(z1.shape[0],device=z1.device)
    return 0.5*(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))

def structural_objective(net,batch,node_ratio=0.2,edge_ratio=0.2,
                         edge_weight=1.,contrast_weight=0.5,consistency_weight=0.1):
    x,edge,valid=batch['x'],batch['edge'],batch['mask']
    nm,me,pairs=create_masks(x,edge,valid,node_ratio,edge_ratio)
    x1=x.clone()
    if pairs[0].numel():
        b,i,j=pairs
        x1[b,j,14]=0.
    h1,z1,score1,selected1=net.encode_view(x1,me,valid,nm)
    pred_nodes=net.node_head(h1)
    regression=F.smooth_l1_loss(pred_nodes[nm][:,CONTINUOUS],x[nm][:,CONTINUOUS])
    target_direction=x[nm][:,11:13].argmax(dim=-1)
    direction=F.cross_entropy(pred_nodes[nm,11:13],target_direction)
    if pairs[0].numel():
        b,i,j=pairs
        pred_edge=net.edge_head(torch.cat((h1[b,i],h1[b,j]),-1)).squeeze(-1)
        temporal=F.smooth_l1_loss(pred_edge,edge[b,i,j,1])
    else:
        temporal=h1.sum()*0.
    second_mask,second_edge,_=create_masks(x,edge,valid,node_ratio,0.)
    h2,z2,score2,selected2=net.encode_view(x,edge,valid,second_mask)
    contrast=graph_info_nce(z1,z2,net.temperature)
    consistency=relation_stability(score1,score2,selected1,selected2)
    loss=regression+direction+edge_weight*temporal+contrast_weight*contrast+consistency_weight*consistency
    stats={'node_reg':regression.detach().item(),'direction':direction.detach().item(),
           'temporal_relation':temporal.detach().item(),'graph_contrast':contrast.detach().item(),
           'relation_consistency':consistency.detach().item(),'selected_pairs':selected1.sum().item()}
    return loss,stats
