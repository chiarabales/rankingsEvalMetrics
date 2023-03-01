import numpy as np
import math
from random import shuffle

import matplotlib.pyplot as plt
import seaborn as sns

import scipy
import scipy.cluster.hierarchy as sch
import pandas as pd
import click
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

@click.command()
@click.option('--_arraylength', type=int)
@click.option('--_trials', type=int)
@click.option('--_relevant', type=int)
@click.option('--_distance', type=str)


def main(_arraylength, _trials, _relevant, _distance):
    
    np.set_printoptions(suppress=True)
    
    #################################################################################
    # _____________________________________________________________________________ #
    
    # __________________________ TO CHANGE DIRECTORY ______________________________ #
    
    _path = pathlib.Path.home() / '21_comparing_rankings/results/robustness/typeIII/'
    _path.mkdir(exist_ok=True)
    
    _file_name1 = f'R_TypeIII_arraylength{_arraylength}_relevant{_relevant}_distance{_distance}_trials{_trials}.txt'
    _file_name2 = f'R_TypeIII_arraylength{_arraylength}_relevant{_relevant}_distance{_distance}_trials{_trials}_metrics.txt'

    output_where1 = _path / _file_name1
    output_where2 = _path / _file_name2
    
    print(f'working on ... _RUN_robustnessIII.py with _arraylength: {_arraylength} and _relevant: {_relevant}')
    
    # _____________________________________________________________________________ #
    
    #################################################################################
    
    __metrics = _utils.metrics(_distance)
    metrics = __metrics[0]
    metrics_names =  __metrics[1]
    
    S = np.arange(1, _arraylength+1)
    R = np.arange(1, _arraylength+1)
    count = np.array([0.0 for i in range(len(metrics))] )
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
    for trail in range(_trials):
        
        j = np.random.randint(_arraylength)
        if j > _arraylength/2:
            k = np.random.randint(j)
        else: 
            k = np.random.randint(j+1, _arraylength)

        shuffle(R)
        shuffle(S)
        Rtilde = _utils.swapElements(R, j, k)
        Stilde = _utils.swapElements(S, j, k)
        
        for met in metrics:
        
            if met not in metrics_withNoRelevance:
                
                _difference = met(S, R, _relevant) - met(Stilde, Rtilde, _relevant)

            else:
                _difference = met(S, R) - met(Stilde, Rtilde)
                     
            m = metrics.index(met)
            
#             print(_difference)
            if np.abs(_difference) >  1e-6:
                count[m] += 1
    
    my_file = open(output_where2, "w")   
    
    for s in range(len(count)):
        if count[s] == 0:
            print(metrics_names[s])
            my_file.write(metrics_names[s] + '\n')
            
    my_file.close()

    count = count*100/_trials
    print(count)

    my_file = open(output_where1, "w")
    np.savetxt(my_file, count.astype('int32'), fmt="%d")
    my_file.close()
    
if __name__ == '__main__':
    main()