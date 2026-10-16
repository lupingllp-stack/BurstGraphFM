"""Global multi-head self-attention with optional edge-feature structural bias."""
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self,d=128,heads=4,dropout=0.1):
        super().__init__()
        self.heads=heads;self.dim=d//heads
        self.norm1=nn.LayerNorm(d);self.norm2=nn.LayerNorm(d)
        self.qkv=nn.Linear(d,3*d);self.out=nn.Linear(d,d)
        self.edge_bias=nn.Linear(4,heads,bias=False)
        self.ffn=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Dropout(dropout),nn.Linear(4*d,d))
        self.drop=nn.Dropout(dropout)
    def forward(self,x,edge,mask,use_edges,extra_bias=None):
        b,n,d=x.shape;z=self.norm1(x)
        q,k,v=self.qkv(z).chunk(3,dim=-1)
        q=q.reshape(b,n,self.heads,self.dim).transpose(1,2)
        k=k.reshape(b,n,self.heads,self.dim).transpose(1,2)
        v=v.reshape(b,n,self.heads,self.dim).transpose(1,2)
        logits=q@k.transpose(-2,-1)*(self.dim**-0.5)
        if use_edges:logits=logits+self.edge_bias(edge).permute(0,3,1,2)
        if extra_bias is not None:logits=logits+extra_bias.permute(0,3,1,2)
        logits=logits.masked_fill(~mask[:,None,None,:],torch.finfo(logits.dtype).min)
        attn=F.softmax(logits,dim=-1)
        h=(self.drop(attn)@v).transpose(1,2).reshape(b,n,d)
        x=x+self.drop(self.out(h));x=x+self.drop(self.ffn(self.norm2(x)))
        return x*mask.unsqueeze(-1)

class BurstClassifier(nn.Module):
    def __init__(self,classes,model_type='graph',hidden=128,layers=4,heads=4,dropout=0.1):
        super().__init__();assert hidden%heads==0
        self.model_type=model_type
        self.input=nn.Linear(16,hidden)
        self.blocks=nn.ModuleList([Block(hidden,heads,dropout) for _ in range(layers)])
        self.head=nn.Sequential(nn.LayerNorm(hidden*2),nn.Linear(hidden*2,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,classes))
    def forward(self,x,edge,mask):
        h=self.input(x)
        for block in self.blocks:h=block(h,edge,mask,self.model_type=='graph')
        denom=mask.sum(1,keepdim=True).clamp_min(1)
        mean=(h*mask.unsqueeze(-1)).sum(1)/denom
        maximum=h.masked_fill(~mask.unsqueeze(-1),torch.finfo(h.dtype).min).max(1).values
        return self.head(torch.cat([mean,maximum],dim=-1))

class StructuralBurstClassifier(BurstClassifier):
    """Fine-tuning retains learned nonlocal relation scorer and attention bias."""
    def __init__(self,classes,hidden=128,layers=4,heads=4,dropout=0.1,top_k=3):
        super().__init__(classes,'graph',hidden,layers,heads,dropout)
        from .interaction import LearnedInteraction
        self.interaction=LearnedInteraction(hidden,heads,top_k=top_k)
    def forward(self,x,edge,mask):
        h=self.input(x)
        bias,_,_,_=self.interaction(h,mask)
        for block in self.blocks:
            h=block(h,edge,mask,True,extra_bias=bias)
        denom=mask.sum(1,keepdim=True).clamp_min(1)
        mean=(h*mask.unsqueeze(-1)).sum(1)/denom
        maximum=h.masked_fill(~mask.unsqueeze(-1),torch.finfo(h.dtype).min).max(1).values
        return self.head(torch.cat([mean,maximum],dim=-1))
