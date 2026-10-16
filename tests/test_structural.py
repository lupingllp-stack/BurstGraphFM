import torch
from burstgraphfm.structural import StructuralPretrainNet,structural_objective
from burstgraphfm.interaction import LearnedInteraction

def make_batch(b=3,n=6):
    torch.manual_seed(9)
    x=torch.randn(b,n,16)
    x[:,:,11]=1.;x[:,:,12]=0.
    edge=torch.zeros(b,n,n,4)
    for k in range(n-1):
        edge[:,k,k+1]=torch.tensor([1.,0.2,1.,1.])
        edge[:,k+1,k]=torch.tensor([1.,0.2,1.,-1.])
    valid=torch.ones(b,n,dtype=torch.bool)
    valid[-1,-2:]=False
    return dict(x=x,edge=edge,mask=valid)

def test_nonlocal_edges_and_gradient():
    batch=make_batch()
    model=StructuralPretrainNet(hidden=32,layers=2,heads=4,dropout=0.,top_k=2)
    loss,stats=structural_objective(model,batch)
    assert torch.isfinite(loss)
    loss.backward()
    grads=[p.grad for p in model.interaction.pair_net.parameters()]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in grads)
    assert model.blocks[0].qkv.weight.grad is not None
    assert stats['selected_pairs']>0
    with torch.no_grad():
        h=model.input(batch['x'])
        bias,chosen,_,eligible=model.interaction(h,batch['mask'])
    assert torch.all(~chosen | eligible)
    assert not chosen[0].diagonal().any()
    for k in range(chosen.shape[1]-1):
        assert not chosen[:,k,k+1].any()
    assert not chosen[-1,-1].any()

def test_single_and_two_node_graphs():
    model=StructuralPretrainNet(hidden=32,layers=1,heads=4)
    batch=make_batch(b=2,n=3)
    batch['mask'][:]=False
    batch['mask'][0,:1]=True
    batch['mask'][1,:2]=True
    loss,_=structural_objective(model,batch)
    assert torch.isfinite(loss)
