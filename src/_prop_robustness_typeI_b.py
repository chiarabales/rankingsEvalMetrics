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

import click

import _utils



@click.command()
@click.option('--_arraylength', type=int)
@click.option('--_trials', type=int)
@click.option('--_distance', type=str)

def main(_arraylength, _trials, _distance):
    
    
    #################################################################################
    # _____________________________________________________________________________ #
    
    # __________________________ TO CHANGE DIRECTORY ______________________________ #
    


    _path = pathlib.Path.home() / '21_comparing_rankings/results/robustness/Robustness1B'
    _path.mkdir(exist_ok=True)
    _file_name = f'ROB1b_2_arraylength_{_arraylength}_trials_{_trials}_distance_{_distance}.txt'
    output_where = _path / _file_name

    
    # _____________________________________________________________________________ #
    
    #################################################################################
    
    
    __metrics = _utils.metrics(_distance)
    metrics = __metrics[0]
    metrics_names =  __metrics[1]
    
    ''' 
    
    we just consider the case we retrieve the same number of elements as the relevant ones
    
    '''
    
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

    S = np.arange(1, _arraylength+1)
    R = np.arange(1, _arraylength+1)

    count = [0 for i in range(len(metrics))]
    dividend = 0
    
    
    for t in range(_trials):
        
        shuffle(R) 
        shuffle(S)

         
        Rtilde = _utils.slide(R) # swap
        Stilde = _utils.slide(S) # swap
        
      
        
    
        for _relevant in range(1,int(_arraylength/2)+1):
            
            
            dividend += 1
            for met in metrics:

                if met not in metrics_withNoRelevance:

                    n = met(S, R, _relevant)
                    ntilde = met(Stilde, R, _relevant)
                    diff_ = np.abs(n - ntilde)

                else:
                    n = met(S, R)
                    ntilde = met(Stilde, R)
                    diff_ = np.abs(n - ntilde)

                m = metrics.index(met)
                
                count[m] += diff_


        count2 = list(np.array(count)/dividend)

        my_file = open(output_where, "w")
        for s in range(len(count2)):
            my_file.write(metrics_names[s] + ':\t' + str(count2[s]) + '\n')
        my_file.close()

    
    
#     count = list(np.array(count)/dividend)

    
#     my_file = open(output_where, "w")
#     for s in range(len(count)):
#         my_file.write(metrics_names[s] + ':\t' + str(count[s]) + '\n')
#     my_file.close()

    
    
    
    
if __name__ == '__main__':
    main()