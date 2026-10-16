import torch
from burstgraphfm.build import make_graph
from burstgraphfm.data import collate
from burstgraphfm.model import BurstClassifier

def test_graph_pipeline():
    pkt=[(0.0,0,100),(0.1,0,120),(0.2,1,90),(0.3,1,80),(0.4,0,130)]
    g=make_graph(pkt)
    assert g['x'].shape==(3,16)
    assert g['edge'].shape==(3,3,4)
    assert g['edge'][0,1,0]==1
    assert torch.isfinite(g['x']).all()
    b=collate([{**g,'y':0},{**g,'y':1}])
    for typ in ['sequence','graph']:
        m=BurstClassifier(2,typ,hidden=32,layers=2,heads=4)
        logits=m(b['x'],b['edge'],b['mask'])
        assert logits.shape==(2,2)
        logits.sum().backward()
