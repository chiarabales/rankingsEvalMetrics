import numpy as np
import math
from random import shuffle

import matplotlib.pyplot as plt
import seaborn as sns

import click

import scipy
import scipy.cluster.hierarchy as sch
import pandas as pd

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
@click.option('--_relevant', type=int)
@click.option('--_trials', type=int)
@click.option('--_distance', type=str)

def main(_arraylength, _relevant, _trials, _distance):
    
    __metrics = _utils.metrics(_distance)
    metrics = __metrics[0]
    metrics_names =  __metrics[1]

    ''' 
    
    we just consider the case we retrieve the same number of elements as the relevant ones
    
    '''

    factorial = math.factorial(_arraylength)
    print("Factorial: " + str(factorial))

    S = np.arange(1, _arraylength+1)
    R1 = np.arange(1, _arraylength+1)
    R2 = np.arange(1, _arraylength+1)

    coincidents = set()
    coincident = False
    
    
    _path = pathlib.Path.home() / '21_comparing_rankings/results/coincident/'
    _path.mkdir(exist_ok=True)
    _file_name = f'coincidents_arraylength{_arraylength}_relevant{_relevant}_trials{_trials}_distance{_distance}.txt'
    output_where = _path / _file_name
    
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
    
    
    for i in range(_trials):

        # shuffle the two rankings
        shuffle(R1)
        shuffle(R2)
        
        for met in metrics:
            
            if met not in metrics_withNoRelevance:
                
                if abs(met(S, R1, _relevant) - met(S, R2, _relevant)) < 1e-6:
                    coincident = True
            else:
                if abs(met(S, R1) - met(S, R2)) < 1e-6:
                    coincident = True
                    
                   
            if coincident == True:
                coincidents.add(met)
                coincident = False
                my_file = open(output_where, "w")
    
                for s in coincidents:
                    my_file.write(str(s) + '\n')
                my_file.close()

        if len(metrics) == 0:
             print('finally finished')

    
if __name__ == '__main__':
    main()