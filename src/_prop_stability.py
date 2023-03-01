import numpy as np
import math
from random import shuffle

import matplotlib.pyplot as plt
import seaborn as sns

import scipy
import scipy.cluster.hierarchy as sch
import pandas as pd

import os
import pathlib

# ___________________________________________

# _________ IMPORT BUILT IN METRICS _________

# ___________________________________________

from scipy.stats import kendalltau as kt
from scipy.stats import spearmanr as spearmanr

# ___________________________________________

# _________ new packages created ____________

# ___________________________________________

import _confusion_matrix as cm
import _metrics_confusion_matrix_based as cmbm
import _metrics_error_based as ebm
import _metrics_correlation_based as cbm
import _metrics_cumulative_gain_based as cgbm
import _metrics_others as om

import _utils

import click



@click.command()
@click.option('--_arraylength', type=int)
@click.option('--_trials', type=int)
@click.option('--_distance', type=str)


def main(_arraylength, _trials, _distance):
    
    
    print(f'working on ... _RUN_stability.py with _arraylength: {_arraylength}')
    
    
    np.set_printoptions(suppress=True)
    
    #################################################################################
    # _____________________________________________________________________________ #
    
    # __________________________ TO CHANGE DIRECTORY ______________________________ #
    
    # _____________________________________________________________________________ #
    
    #################################################################################
    
    __metrics = _utils.metrics(_distance)
    metrics = __metrics[0]
    metrics_names =  __metrics[1]
    

    S = np.arange(1, _arraylength + 1)
    R = np.arange(1, _arraylength + 1)
    
    metrics_withNoRelevance = [cbm.ktau
                               , cbm.srho
                               , cgbm.dist_dcg, cgbm.distAbs_dcg
                               , cgbm.dist_ndcg, cgbm.distAbs_ndcg
                               , om.NDPM
                               , ebm.mse
                               , ebm.rmse
                               , ebm.mae
                               , ebm.mape
                               , ebm.smape
                               , ebm.r2score]
    
    stability = np.array([[[0.0 for _relevant in range(_arraylength)] for t in range(_trials)] for i in range(len(metrics))])        
    shuffle(R)
    
    for met in metrics:
        for _trial in range(_trials):
            shuffle(R)
            for _relevant in range(1,_arraylength):

                if met not in metrics_withNoRelevance:

                    n = met(S, R, _relevant)
                else:
                    n = met(S, R)

                m = metrics.index(met)
                stability[m][_trial][_relevant] = n
                
        _path = pathlib.Path.home() / f'21_comparing_rankings/results/stability/{metrics_names[m]}/'
        _path.mkdir(exist_ok=True)
        _file_name = f'stability_arraylength{_arraylength}_trial{_trial}_distance{_distance}.txt'
        output_where = _path / _file_name

        my_file = open(output_where, "w")
        np.savetxt(my_file, stability[m], delimiter=',', fmt='%f')
        my_file.close()

    return stability

    
if __name__ == '__main__':
    main()