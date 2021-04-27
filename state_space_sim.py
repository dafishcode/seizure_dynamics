## Functions 
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

def rand_project(n_samples, n_features, n_components, alpha, theta):
    W = np.random.randn(n_components, n_features) #weight matrix - entirely random - 3 PCs
    X = np.zeros((n_features, n_samples)) #x matrix - features x samples

    for n in range(1, (n_features//2)):
        X[(2*n)-1,:] = np.cos(theta*n)/n**(alpha/2) 
        X[(2*n), :] = np.sin(theta*n)/n**(alpha/2) 

    #for n in range(1, (n_features)):
    #  X[n] = np.sin(n*theta)/n**(alpha/2)


    wproj = W @ X
    return(wproj, W, X)

def eigenspec(scores):
    var_vec = np.zeros(len(scores))
    for i in range(len(scores)):
        var_vec[i] = np.var(scores[i])
        eigvar = np.zeros(len(scores))  
    for i in range(len(scores)):
        eigvar[i] = var_vec[i]/np.sum(var_vec)

    return(eigvar)
