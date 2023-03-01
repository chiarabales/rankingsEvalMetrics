from scipy.stats import gmean
import itertools
import numpy as np

# ______________________________________________________________________________


def rr(_element, rank2):
    for i, item in enumerate(rank2):
        if item == _element:
            return 1.0 / (i + 1.0)
    raise ValueError('element non recommended')
    
    
def MRR(rank1, rank2, j):

    # DEFINITION OF MRR
    
    _rel = rank1[:j]
    mrr = 0
    for _element in _rel:
        mrr += rr(_element, rank2)
    mrr = mrr/len(rank1)
    return mrr

# ______________________________________________________________________________


def rank(_element, rank2):
    
    for i, item in enumerate(rank2):
        if item == _element:
            return i + 1 
    raise ValueError('element non recommended')
    
    
def meanRank(rank1, rank2, j):
    
    # DEFINITION OF MEAN RANK
    
    rank1 = rank1[:j]
    
    mr = 0
    for _element in rank1:
        mr += rank(_element, rank2)
    mr = mr/len(rank1)
    return mr


def GMR(rank1, rank2, j):
    
    # CALCULATE THE GEOMETRIC MEAN
    
    rank1 = rank1[:j]
    array = [rank(_element, rank2) for _element in rank1]
    gmeanRank = gmean(array)
    
    return gmeanRank

# ______________________________________________________________________________


def HitsAtK(rank1, rank2, k = 1000):
    
    if len(rank2) > k:
        rank2 = rank2[:k]
    h = 0
    for i, item in enumerate(rank2):
        if item in rank1 and item not in rank2[:i]:
            h += 1
    h = h / len(rank2)
    
    return h

# ______________________________________________________________________________


def NDPM(rank1, rank2):
    
    # code is not mine
    
    """
    Calculates the Normalized Distance-based Performance Measure (NPDM) between two
    ordered lists. Two matching orderings return 0.0 while two unmatched orderings returns 1.0.
    Args:
        relevant_items (List): List of items
        recommendation (List): The predicted list of items
    Returns:
        NDPM (float): Normalized Distance-based Performance Measure
    Metric Definition:
    Yao, Y. Y. "Measuring retrieval effectiveness based on user preference of documents."
    Journal of the American Society for Information science 46.2 (1995): 133-145.
    Definition from:
    Shani, Guy, and Asela Gunawardana. "Evaluating recommendation systems."
    Recommender systems handbook. Springer, Boston, MA, 2011. 257-297
    """
    assert set(rank1) == set(rank2)

    _items_rank = {item: i + 1 for i, item in enumerate(dict.fromkeys(rank1))}
    _predicted_rank = {item: i + 1 for i, item in enumerate(dict.fromkeys(rank2))}

    items = set(rank1)

    _combinations = itertools.combinations(items, 2)

    C_minus = 0
    C_plus = 0
    C_u = 0

    for item1, item2 in _combinations:
        _items_rank1 = _items_rank[item1]
        _items_rank2 = _items_rank[item2]

        item1_pred_rank = _predicted_rank[item1]
        item2_pred_rank = _predicted_rank[item2]

        C = np.sign(item1_pred_rank - item2_pred_rank) * np.sign(_items_rank1 - _items_rank2)

        C_u += C ** 2

        if C < 0:
            C_minus += 1
        else:
            C_plus += 1

    C_u0 = C_u - (C_plus + C_minus)

    NDPM = (C_minus + 0.5 * C_u0) / C_u
    return NDPM