from scipy.stats import kendalltau as kt
from scipy.stats import spearmanr as spearmanr

def ktau(rank1, rank2):
    tau = kt(rank1, rank2)[0]
    return tau

def srho(rank1, rank2):
    r = spearmanr(rank1, rank2)[0]
    return r




