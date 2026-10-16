import torch
from burstgraphfm.pretrain import PretrainNet, objective, create_masks

def test_ssl_gradient_and_masked_edges():
    torch.manual_seed(3)
    x=torch.randn(2,4,16)
    x[:,:,11]=1.;x[:,:,12]=0.
    edge=torch.zeros(2,4,4,4)
    for i in range(3):
        edge[:,i,i+1]=torch.tensor([1.,0.3,1.,1.])
        edge[:,i+1,i]=torch.tensor([1.,0.3,1.,-1.])
    valid=torch.ones(2,4,dtype=torch.bool)
    nodes, masked, pairs=create_masks(x,edge,valid,0.25,0.5)
    assert nodes.any() and pairs[0].numel()>0
    assert (masked[pairs[0],pairs[1],pairs[2]]==0).all()
    net=PretrainNet(hidden=32,layers=2,heads=4)
    loss,_=objective(net,dict(x=x,edge=edge,mask=valid))
    assert torch.isfinite(loss)
    loss.backward()
    assert net.input.weight.grad is not None
    assert net.edge_head[-1].weight.grad is not None

def test_single_burst_without_edge():
    net=PretrainNet(hidden=32,layers=1,heads=4)
    x=torch.zeros(2,1,16)
    x[:,:,11]=1.
    loss,_=objective(net,dict(x=x,edge=torch.zeros(2,1,1,4),mask=torch.ones(2,1,dtype=torch.bool)))
    assert torch.isfinite(loss)
    loss.backward()
