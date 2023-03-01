import numpy as np

from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def mse(rank1, rank2):
    error = mean_squared_error(rank1, rank2)
    return error

def rmse(rank1, rank2):
    error = np.sqrt(mse(rank1, rank2))
    return error

def mae(rank1, rank2):
    error = mean_absolute_error(rank1, rank2)
    return error

# def mape(rank1, rank2):
#     # NON SONO SICURA DI QUESTA
# #     mape = np.mean(np.abs((rank1 - rank2)/rank1))*100
# #     mape = mean_absolute_percentage_error(rank1, rank2)
#     return True

# def smape(rank1, rank2):
#     # NON SONO SICURA DI QUESTA
#     error = 100/len(rank1) * np.sum(2 * np.abs(rank2 - rank1) / (np.abs(rank1) + np.abs(rank2)))
#     return error

def mape(rank1,rank2):
    MAPE = np.mean(np.abs((rank1 - rank2)/rank1))*100
    return MAPE


def smape(rank1, rank2):
    error = 100/len(rank1) * np.sum(2 * np.abs(rank2 - rank1) / (np.abs(rank1) + np.abs(rank2)))
    return error
    

def SMAPE2(predicted, actual):
    return np.mean(
            np.abs(predicted - actual) / 
            ((np.abs(predicted) + np.abs(actual))/2))*100

def r2score(rank1, rank2):
    error = r2_score(rank1, rank2)
    return error



# def pearson(rank1, rank2):
#     error = pearsonr(rank1, rank2)[0]
#     return error
