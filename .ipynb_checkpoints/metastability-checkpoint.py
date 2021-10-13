#Cluster with affinity propagation
#==============================
def affprop(data):
#==============================
    """
    This function performs affinity propagation on state vectors. 
    
    Inputs:
        data (np array): cells x timepoints, state vectors
        
    Returns:
        all_c (np array): 1d vector of cluster labels for each time point
        sub_c (np array): 1d vector of all unique cluster labels, that label more than a single time point

    """
    from sklearn.cluster import AffinityPropagation
    import numpy as np
    
    
    cluster = AffinityPropagation(damping = 0.5, max_iter = 200, convergence_iter = 15).fit(data)
    unq,counts = np.unique(cluster.labels_, return_counts = True)
    all_c = cluster.labels_
    sub_c = unq[counts > 1] #Remove clusters that have only a singular member
    return(all_c, sub_c)

#Similarity
#==============================
def Similarity(curr_clust):
#==============================
    """
    This function calculates the mean similarity between state vecotrs belonging to a cluster.
    
    Inputs:
        curr_clust (np array): all state vectors belonging to this cluster
        
    Returns:
        mean_sim (float): the mean similarity

    """
    import numpy as np
    
    ijdot = np.inner(curr_clust, curr_clust)
    self_dot = np.apply_along_axis(np.max,0,ijdot)
    idot = np.reshape(np.repeat(self_dot, ijdot.shape[0]), ijdot.shape)
    jdot = np.reshape(np.repeat(self_dot, ijdot.shape[0]), ijdot.shape).T
    sim_mat = np.triu(ijdot / (idot + jdot - ijdot))
    np.fill_diagonal(sim_mat,0)
    mean_sim = np.mean(sim_mat[np.nonzero(sim_mat)])
    return(mean_sim)

#=========================================
def Sim_loop(data, all_clust, sub_clust):
#==========================================
    """
    This function loops through all clusters in a dataset and finds the mean similarity for each cluster. 
    
    Inputs:
        data (np array): cells x timepoints
        all_clust (np array): 1d vector of cluster labels for each time point
        sub_clust (np array): 1d vector of all unique cluster labels, that label more than a single time point

        
    Returns:
        sim_list (list): list of all similarities for each cluster

    """
    import numpy as np
    
    sim_list = list(range(len(sub_clust)))
    
    #Loop through all clusters with more than 1 member
    for i in range(len(sub_clust)):
        curr_clust = data[np.where(all_clust == sub_clust[i])[0]] #Find all time frames belonging to current cluster
        sim_list[i] = Similarity(curr_clust) #Calculate mean similarity for this cluster
    return(sim_list)


#==========================================
def state_stats(fin_clust, all_clust):
#==========================================
    """
    This function calculates the probability and mean dwell times of each state. 
    
    Inputs:
        fin_clust (np array): 1d vector of state cluster labels occur with above chance probability
        all_clust (np array): 1d vector of cluster labels for each time point
        
    Returns:
        p_state (np.array):  1d vec containing probabilities of each state
        m_dwell (np.array): 1d vec containing the mean dwell time for each state
        full_vec (list): contains all durations in between every single state transition 
        
    """

    import more_itertools as mit
    import numpy as np

    p_state, m_dwell = np.zeros(len(fin_clust)),np.zeros(len(fin_clust)) 
    
    full_vec = list(range(len(fin_clust)))
    for i in range(len(fin_clust)):
        
        #calculate probabilities of each state
        p_state[i] = len(np.where(all_clust == fin_clust[i])[0])/len(all_clust)
        
        #find all periods with the same state over consecutive time frames
        dur_list = [list(group) for group in mit.consecutive_groups(np.where(all_clust == fin_clust[i])[0])]
        vec = []
        
        for t in range(len(dur_list)):
            vec = np.append(vec, len(dur_list[t]))
        m_dwell[i] = np.mean(vec)
        full_vec[i] = vec
    return(p_state, m_dwell, full_vec)


#==========================================
def null_states(fin_clust, data):
#==========================================
    """
    This function calculates the mean dwell time in a system with a given number of states and random dynamics. 
    
    Inputs:
        fin_clust (np array): 1d vector of state cluster labels occur with above chance probability
        data (np array): cells x timepoints
        
    Returns:
        null_m_dwell (np.array): 1d vec containing the mean dwell time for each state

    """

    import random
    import more_itertools as mit
    import numpy as np

    all_states = np.arange(1,len(fin_clust)+1)
    rand_states = np.array(random.choices(all_states, k = data.shape[0]))
    dur_list = [list(group) for group in mit.consecutive_groups(rand_states)]
    vec = []
    for t in range(len(dur_list)):
        vec = np.append(vec, len(dur_list[t]))
    null_m_dwell = np.mean(vec)
    return(null_m_dwell)


