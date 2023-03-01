import numpy as np

# ___________________________________________________


import _confusion_matrix as cm
import _metrics_confusion_matrix_based as cmbm
import _metrics_error_based as ebm
import _metrics_correlation_based as cbm
import _metrics_cumulative_gain_based as cgbm
import _metrics_others as om


def metrics(distance = None):
    
    _metrics = [
        
        # confusion matrix based metrics
        cmbm.recall
        , cmbm.fnr
        , cmbm.fallout
        , cmbm.tnr
        , cmbm.precision
        , cmbm.fdr
        , cmbm.npv
        , cmbm.FOR
        , cmbm.accuracy
        , cmbm.ba
        , cmbm.f1_score
        , cmbm.fm
        , cmbm.mcc
        , cmbm.jaccard_index
        , cmbm.lr_plus
        , cmbm.lr_minus
        , cmbm.informedness
        , cmbm.pt
        , cmbm.markedness
        
        # error based metrics
        , ebm.mse
        , ebm.rmse
        , ebm.mae
        , ebm.mape
        , ebm.smape
        , ebm.r2score
        
        # correlation based metrics
        , cbm.ktau
        , cbm.srho
        , om.NDPM
        
        # cumulative gain based metrics
        , cgbm.dcg
        , cgbm.ndcg
        , om.MRR
        , om.GMR
        , om.meanRank
        
    ]

    _metrics_names = [
        
        # confusion matrix based metrics
        'recall'
        , 'FNR'
        , 'fallout'
        , 'TNR'
        , 'precision'
        , 'FDR'
        , 'NPV'
        , 'FOR'
        , 'accuracy'
        , 'BA'
        , 'F1 score'
        , 'FM'
        , 'MCC'
        , 'Jaccard index'
        , 'LR+'
        , 'LR-'
        , 'informedness'
        , 'PT'
        , 'markedness'
        
        # error based metrics
        , 'MSE'
        , 'RMSE'
        , 'MAE'
        , 'MAPE'
        , 'SMAPE'
        , 'R2 score'
        
        # correlation based metrics
        , 'Kendall tau'
        , 'Spearmann rho'
        , 'NDPM'
        
        # cumulative gain based metrics
        , 'DCG'
        , 'nDCG'
        , 'MRR'
        , 'GMR'
        , 'meanRank'
        
    ]
    
    if distance == 'normal':
        _metrics[28] = cgbm.dist_dcg
        _metrics[29] = cgbm.dist_ndcg
        
    elif distance == 'abs':
        _metrics[28] = cgbm.distAbs_dcg
        _metrics[29] = cgbm.distAbs_ndcg
        
    return _metrics, _metrics_names

##################################################################################

# ________________________________________________________________________________

def swapElementsAdjacent(list, n):
    
    # function that swaps two elements close to each other
    # _______________________________________________________

    list[n], list[n+1] = list[n+1], list[n]
    return list

def swapElements(array, n, m):
    
    S = np.arange(len(array))
   
    
    for i in range(len(array)):
        if i == n:
            S[i] = array[m]
        elif i == m:
            S[i] = array[n]
        else:
            S[i]= array[i]

    # function that swaps two elements of an array
    # _______________________________________________________
    
    return S


def slide(array):
    
    # function that slides the entire array 
    # _______________________________________________________

    slidedarray = np.arange(len(array))
    for i in range(len(array)-1):
        slidedarray[i] = array[i+1]
    slidedarray[len(array)-1] = array[0]
    return slidedarray
