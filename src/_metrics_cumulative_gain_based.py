from sklearn.metrics import dcg_score as dcg_score
from sklearn.metrics import ndcg_score as ndcg_score

import numpy as np


def dcg(rank):
    
    # note that takes only one argument (only one array)
    
    _arraylength = len(rank)
    rank = np.reshape(rank, (1, _arraylength))
    R = np.arange(1, _arraylength + 1)[::-1]
    R = np.reshape(R, (1, _arraylength))
    
    return dcg_score(R, rank)

def ndcg(rank):
    
    _arraylength = len(rank)
    rank = np.reshape(rank, (1, _arraylength))
    R = np.arange(1, _arraylength + 1)[::-1]
    R = np.reshape(R, (1, _arraylength))
    
    return ndcg_score(R, rank)

# induced distances from the metrics DCG and nDCG

def dist_dcg(rank1, rank2):
    return dcg(rank1) - dcg(rank2)

def dist_ndcg(rank1, rank2):
    return ndcg(rank1) - ndcg(rank2)

def distAbs_dcg(rank1, rank2):
    return abs(dcg(rank1) - dcg(rank2))

def distAbs_ndcg(rank1, rank2):
    return abs(ndcg(rank1) - ndcg(rank2))