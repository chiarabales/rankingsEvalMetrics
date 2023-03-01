import numpy as np

def relevant_elements(rank, j):
    relevant = set(rank[:j])
    return relevant

def retrieved_elements(rank, j):
    retrieved = set(rank[:j])
    return retrieved

def non_relevant_elements(rank, j):
    relevant = relevant_elements(rank, j)
    all_elements = set(rank)
    non_relevant = all_elements.difference(relevant)
    return non_relevant

def non_retrieved_elements(rank, j):
    retrieved = retrieved_elements(rank, j)
    all_elements = set(rank)
    non_retrieved = all_elements.difference(retrieved)
    return non_retrieved

def TP(rank1, rank2, j):
    retrieved = retrieved_elements(rank2, j)
    relevant = relevant_elements(rank1, j)
    tp = retrieved.intersection(relevant)
    return len(tp)

def FP(rank1, rank2, j):
    retrieved = retrieved_elements(rank2, j)
    relevant = relevant_elements(rank1, j)
    fp =  retrieved.difference(relevant)
    return len(fp)

def TN(rank1, rank2, j):
    non_retrieved = non_retrieved_elements(rank2, j)
    non_relevant = non_relevant_elements(rank1, j)
    tn = non_retrieved.intersection(non_relevant)
    return len(tn)

def FN(rank1, rank2, j):
    non_retrieved = non_retrieved_elements(rank2, j)
    non_relevant = non_relevant_elements(rank1, j)
    fn = non_retrieved.difference(non_relevant)
    return len(fn)