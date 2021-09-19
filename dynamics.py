#=======================================================================================
def eigspec(data, n_components):
#======================================================================================
    """
    This functions calculates the eigenspectrum over an array. The eigenspectrum is the amount of variance that each PC captures. 
    
    Inputs:
        data_list(np array): cells x time frames
        n_components (int): number of components to perform PCA with

    Returns:
        spec (np array): 1d vector of variance ratio captured by each PC from 1 onwards
    """

    from sklearn import decomposition
    pca = decomposition.PCA(n_components)
    fit = pca.fit(data)
    spec = fit.explained_variance_ratio_
    return(spec)


#======================================================================================
def rand_project(n_samples, n_features, n_components, alpha, theta):
#======================================================================================
    """
    This functions simulates eigenspectra power law, by calculating PCs which reduce in variance as theta increases. It then projects them randomly into a n dimensional space for visualisation. 
    
    Inputs:
        n_samples (int): number of time frames for each PC
        n_features (int): number of PCs
        n_components (int): number of components to randomly projects PC matrix into for visualisation
        alpha (float): eigenpspectrum exponent
        theta (float): parameter that defines the amount of variance that each subsequent PC captures

    Returns:
        wproj (np array): n_components x n_samples, random projection of all PCs
        W (np array): n_components x n_features, random weight matrix for projection
        X (np array): n_features x n_samples, full PC matrix of PCs over time
    """
    
    import numpy as np
    
    W = np.random.randn(n_components, n_features) #weight matrix - entirely random - for random projection
    X = np.zeros((n_features, n_samples)) #x matrix - features x samples

    for n in range(1, (n_features//2)):
        X[(2*n)-1,:] = np.cos(theta*n)/n**(alpha/2) 
        X[(2*n), :] = np.sin(theta*n)/n**(alpha/2) 

    #for n in range(1, (n_features)):
    #  X[n] = np.sin(n*theta)/n**(alpha/2)

    wproj = W @ X #matrix multiplication ensures mixing of all PCs into wproj random projection - when early PCs have greater variance, these PCs will drive most of the variance in wproj - this allows smoother trajectories in state space
    
    return(wproj, W, X)

#======================================================================================
def eigspec_sim(X):
#======================================================================================
    """
    This function calculates the variance of each PC from the simulated PCs. 
    
    Inputs:
        X (np array): PCs x samples

    Returns:
        eig_var (np array):  1d vector of variance ratio captured by each PC from 1 onwards
    """
    
    import numpy as np
    var_vec = np.zeros(len(X))
    for i in range(len(X)):
        var_vec[i] = np.var(X[i])
        eigvar = np.zeros(len(X))  
    for i in range(len(X)):
        eigvar[i] = var_vec[i]/np.sum(var_vec)

    return(eigvar)

#======================================================================================
def nonlinembed(data):
#======================================================================================
    """
    This function performs isomap embedding on data. 
    
    Inputs:
        data (np array)

    Returns:
        X_transformed(np array): embedded data
    """
    from sklearn.manifold import Isomap

    embedding = Isomap(n_components=3)
    X_transformed = embedding.fit_transform(data)
    return(X_transformed)

#======================================================================================
def state_dist(data):
#======================================================================================
    """
    This functions calculates the euclidean distance from one point in time to to the next in state space. 
    
    Inputs:
        data (np array): cells x timeframes

    Returns:
        dist (np array): 1d vector, distance distribution
    """
    import numpy as np
    dist = np.zeros((data.shape[1])-1)
    for i in range(dist.shape[0]): 
        dist[i] = np.linalg.norm(data[:,i] - data[:,i+1])#euclidean distance distribution
    return(dist)