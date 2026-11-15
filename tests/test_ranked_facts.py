from inference_scaling.retrieval.ranked_facts import FactRanking,rank_windows


def test_parent_fusion_can_recover_first_parent_and_cap_duplicate_windows():
    parents=['target','other']
    windows=['other@0-100','other@50-150','target@0-100']
    ranked,_=rank_windows(parents,windows,FactRanking(parent_weight=0),set())
    assert ranked==windows
    ranked,_=rank_windows(parents,windows,FactRanking(parent_weight=.8,max_windows_per_parent=1),set())
    assert ranked==['target@0-100','other@0-100']
    ranked,_=rank_windows(parents,windows,FactRanking(parent_weight=.8,max_windows_per_parent=1),{'target@0-100'})
    assert ranked==['other@0-100']


def test_parent_cap_zero_keeps_windows_and_exclusion_does_not_rerank():
    p=FactRanking(parent_weight=0)
    ranked,scores=rank_windows(['a','b'],['b@0-2','a@0-2','a@1-3'],p,{'b@0-2'})
    assert ranked==['a@0-2','a@1-3']
    assert scores['a@0-2']==1/22
