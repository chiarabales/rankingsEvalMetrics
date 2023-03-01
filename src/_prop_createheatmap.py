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


def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]

@click.command()
@click.option('--_arraylength', type=int)
@click.option('--_relevant', type=int)
@click.option('--_trials', type=int)
@click.option('--_distance', type=str)



def main(_arraylength, _relevant, _trials, _distance = 'normal'):
    
    
    #################################################################################
    # _____________________________________________________________________________ #
    
    # __________________________ TO CHANGE DIRECTORY ______________________________ #
    


    _path = pathlib.Path.home() / '21_comparing_rankings/plots/heatmap/'
    _path.mkdir(exist_ok=True)
    _file_name = f'heatmap_arraylength{_arraylength}_relevant{_relevant}_trials{_trials}_distance{_distance}.pdf'
    output_where = _path / _file_name

    
    # _____________________________________________________________________________ #
    
    #################################################################################

    _distance = 'normal'
    __metrics = _utils.metrics(_distance)
    print(__metrics)
    metrics = __metrics[0]
    metrics_names =  __metrics[1]
    
                 
    factorial = math.factorial(_arraylength)
    print("Factorial: " + str(factorial))

    S = np.arange(1, _arraylength+1)
    R1 = np.arange(1, _arraylength+1)
    R2 = np.arange(1, _arraylength+1)


    count = [[0 for i in range(len(metrics))] for i in range(len(metrics))]
    
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
    
    metrics_ToSwap = [cmbm.lr_minus, cmbm.fallout, cmbm.fdr, cmbm.FOR, cmbm.fnr, cmbm.pt, om.NDPM]


    for i in range(_trials):

        # shuffle the two rankings
        shuffle(R1)
        shuffle(R2)

        if i % 100 == 0:
            print(i)

        for met1 in metrics:
            if met1 not in metrics_withNoRelevance:
                
                m1 = met1(S, R1, _relevant)
                m2 = met1(S, R2, _relevant)
                
            else:
                m1 = met1(S, R1)
                m2 = met1(S, R2)
                
            for met2 in metrics:
                if met2 not in metrics_withNoRelevance:
                    
                    n1 = met2(S, R1, _relevant)
                    n2 = met2(S, R2, _relevant)
                    
                else:
                    n1 = met2(S, R1)
                    n2 = met2(S, R2)
            
                m = metrics.index(met1)
                n = metrics.index(met2)

                if ((met1 not in metrics_ToSwap and met2 not in metrics_ToSwap) or (met1 in metrics_ToSwap and met2 in metrics_ToSwap)):
                    if ((m1 >= m2 and n1 >= n2) or (m1 <= m2 and n1 <= n2)):
                        count[m][n] += 1 
                else: 
                    if ((m1 >= m2 and n1 <= n2) or (m1 <= m2 and n1 >= n2)):
                        count[m][n] += 1 
                        
    count2 = [[i/_trials for i in count[j]] for j in range(np.shape(count)[0])]


    df = pd.DataFrame(count2,
                     index = metrics_names, 
                     columns = metrics_names)

    clustered_df = cluster_corr(df)
    fig, ax = plt.subplots(figsize=(12, 10))

    cmap_reversed = plt.cm.get_cmap('Blues_r')
    heatmap = ax.pcolor(clustered_df, cmap=cmap_reversed)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(clustered_df.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(clustered_df.shape[0]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(clustered_df.columns, minor=False, rotation = 90, fontsize = 18)
    ax.set_yticklabels(clustered_df.columns, minor=False, fontsize = 18)

    fig.colorbar(heatmap, ax=ax)
    
    plt.savefig(output_where, bbox_inches = 'tight')
    plt.show()
    
    
if __name__ == '__main__':
    main()
