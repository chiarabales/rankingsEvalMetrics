import _confusion_matrix as cm
import numpy as np

def recall(rank1, rank2, j):
    
    # TRUE POSITIVE RATE or RECALL
    
    tp = cm.TP(rank1, rank2, j)
    fp = cm.FP(rank1, rank2, j)
    tn = cm.TN(rank1, rank2, j)
    fn = cm.FN(rank1, rank2, j)
    
    recall = tp/(tp + fn)
    
    return recall

# ________________________________________________________


def fnr(rank1, rank2, j):
    
    # FALSE NEGATIVE RATE
    
    tpr = recall(rank1, rank2, j)
    FNR = 1 - tpr
    
    return FNR

# ________________________________________________________

def fallout(rank1, rank2, j):
    
    # or FALSE POSITIVE RATE
    
    tp = cm.TP(rank1, rank2, j)
    fp = cm.FP(rank1, rank2, j)
    tn = cm.TN(rank1, rank2, j)
    fn = cm.FN(rank1, rank2, j)
    
    fallout = fp/(fp+tn)
    
    return fallout

# ________________________________________________________


def tnr(rank1, rank2, j):
    
    # FALSE NEGATIVE RATE
    
    fpr = fallout(rank1, rank2, j)
    TNR = 1 - fpr
    
    return TNR

# ________________________________________________________


def precision(rank1, rank2, j):
    
    # or POSITIVE PREDICTED VALUE
    
    tp = cm.TP(rank1, rank2, j)
    fp = cm.FP(rank1, rank2, j)
    tn = cm.TN(rank1, rank2, j)
    fn = cm.FN(rank1, rank2, j)
    
    precision = tp/(tp+fp)
    
    return precision

# ________________________________________________________


def fdr(rank1, rank2, j):
    
    # FALSE DISCOVERY RATE
    
    ppv = precision(rank1, rank2, j)
    FDR = 1 - ppv
    
    return FDR

# ________________________________________________________


def npv(rank1, rank2, j):
    
    # or NEGATIVE PREDICTED VALUE
    
    tp = cm.TP(rank1, rank2, j)
    fp = cm.FP(rank1, rank2, j)
    tn = cm.TN(rank1, rank2, j)
    fn = cm.FN(rank1, rank2, j)
    
    NPV = tn/(fn+tn)
    
    return NPV

# ________________________________________________________


def FOR(rank1, rank2, j):
    
    # or FALSE OMISSION RATE
    
    NPV = npv(rank1, rank2, j)
    f = 1 - NPV
    
    return f

# ________________________________________________________


def accuracy(rank1, rank2, j):
        
    tp = cm.TP(rank1, rank2, j)
    fp = cm.FP(rank1, rank2, j)
    tn = cm.TN(rank1, rank2, j)
    fn = cm.FN(rank1, rank2, j)
    
    accuracy = (tp+tn)/(tp + fn + tn + fp)
    
    return accuracy

# ________________________________________________________


def ba(rank1, rank2, j):
    
    # or BALANCED ACCURACY
    
    tp = cm.TP(rank1, rank2, j)
    fp = cm.FP(rank1, rank2, j)
    tn = cm.TN(rank1, rank2, j)
    fn = cm.FN(rank1, rank2, j)
    
    accuracy = (tp + tn)/(tp + fn + tn + fp)
    
    return accuracy


# ________________________________________________________


def f1_score(rank1, rank2, j):
    
    tp = cm.TP(rank1, rank2, j)
    fp = cm.FP(rank1, rank2, j)
    tn = cm.TN(rank1, rank2, j)
    fn = cm.FN(rank1, rank2, j)
    
    f1_score = 2*tp/(2*tp+fp+fn)
    
    return f1_score

# ________________________________________________________


def fm(rank1, rank2, j):
    
    # or FOWLKES-MALLOWS INDEX
    
    ppv = precision(rank1, rank2, j)
    tpr = recall(rank1, rank2, j)
    
    fm = np.sqrt(ppv*tpr)
    
    return fm

# ________________________________________________________


def mcc(rank1, rank2, j):
    
    # or FOWLKES-MALLOWS INDEX
    
    tpr = recall(rank1, rank2, j)
    TNR = tnr(rank1, rank2, j)
    ppv = precision(rank1, rank2, j)
    NPV = npv(rank1, rank2, j)
    FNR = fnr(rank1, rank2, j)
    fpr = fallout(rank1, rank2, j)
    f = FOR(rank1, rank2, j)
    FDR = fdr(rank1, rank2, j)
    
    mcc = np.sqrt(tpr*TNR*ppv*NPV) - np.sqrt(FNR*fpr*f*FDR)
    
    return mcc

# ________________________________________________________

def jaccard_index(rank1, rank2, j):
    
    tp = cm.TP(rank1, rank2, j)
    fp = cm.FP(rank1, rank2, j)
    tn = cm.TN(rank1, rank2, j)
    fn = cm.FN(rank1, rank2, j)
    
    jaccard = tp / (tp + fn + fp)
    
    return jaccard

# ________________________________________________________

def lr_plus(rank1, rank2, j):
    
    # POSITIVE LIKELIHOOD RATIO
    
    tpr = recall(rank1, rank2, j)
    fpr = fallout(rank1, rank2, j)
    
    if fpr != 0:
        lr = tpr/fpr
    else:
        return -1
    
    return lr

# ________________________________________________________

def lr_minus(rank1, rank2, j):
    
    # NEGATIVE LIKELIHOOD RATIO
    
    FNR = fnr(rank1, rank2, j)
    TNR = tnr(rank1, rank2, j)
    
    if TNR != 0:
        lr = FNR/TNR
    else:
        return -1
    
    return lr
# ________________________________________________________


def informedness(rank1, rank2, j):
    
    # POSITIVE LIKELIHOOD RATIO
    
    TPR = recall(rank1, rank2, j)
    TNR = tnr(rank1, rank2, j)
    
    i = TPR + TNR - 1
    
    return i

# ________________________________________________________

def pt(rank1, rank2, j):
    
    # or PREVALENCE THRESHOLD
    
    tpr = recall(rank1, rank2, j)
    fpr = fallout(rank1, rank2, j)
    
    if tpr != fpr:
        pt = (np.sqrt(tpr*fpr) - fpr)/(tpr-fpr)
    else:
        return -1
    
    return pt

# ________________________________________________________

def prevalence(rank1, rank2, j):
    
    '''
    
    defined as P/(P+N)
    in our case, j = P
    
    '''
    
    score = j/len(rank1)
    
    return score

# ________________________________________________________

def markedness(rank1, rank2, j):
    
    PPV = precision(rank1, rank2, j)
    NPV = npv(rank1, rank2, j)
    
    return PPV + NPV - 1

# ________________________________________________________


def dor(rank1, rank2, j):
    
    lr_p = lr_plus(rank1, rank2, j)
    lr_m = lr_minus(rank1, rank2, j)
    
    return lr_p/lr_m