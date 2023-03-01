from sklearn.metrics import dcg_score as dcg_score
from sklearn.metrics import ndcg_score as ndcg_score

import numpy as np

# def transform_ranking(rank1, rank2):
#     max1 = len(rank1)
#     max2 = len(rank2)
#     n = len(np.arange(np.min([np.min(rank1), np.min(rank2)]), np.max([np.max(rank1), np.max(rank2)])+1))
    
#     y_rank1 = np.zeros(shape=(max1, n))
#     y_rank2 = np.zeros(shape=(max2, n))
    
#     y_rank1[np.arange(max1), rank1] = 1
#     y_rank2[np.arange(max2), rank2] = 1
    
#     return y_rank1, y_rank2

# def dcg(rank1, rank2):
    
#     y_rank1, y_rank2 = transform_ranking(rank1, rank2)
#     dcg_coef = dcg_score(y_rank1, y_rank2)
    
#     return dcg_coef

# def ndcg(rank1, rank2):
    
#     y_rank1, y_rank2 = transform_ranking(rank1, rank2)
#     ndcg_coef = ndcg_score(y_rank1, y_rank2)
    
#     return ndcg_coef


def dcg(rank):
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

def dist_dcg(rank1, rank2):
    return dcg(rank1) - dcg(rank2)

def dist_ndcg(rank1, rank2):
    return ndcg(rank1) - ndcg(rank2)

def distAbs_dcg(rank1, rank2):
    return abs(dcg(rank1) - dcg(rank2))

def distAbs_ndcg(rank1, rank2):
    return abs(ndcg(rank1) - ndcg(rank2))