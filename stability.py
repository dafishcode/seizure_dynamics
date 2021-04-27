#Cluster with affinity propagation
def affprop(data):
    cluster = AffinityPropagation(damping = 0.5, max_iter = 200, convergence_iter = 15).fit(data)
    unq,counts = np.unique(cluster.labels_, return_counts = True)
    all_c = cluster.labels_
    sub_c = unq[counts > 1]
    return(all_c, sub_c)

#Similarity
def Similarity(curr_clust):
    ijdot = np.inner(curr_clust, curr_clust)
    self_dot = np.apply_along_axis(np.max,0,ijdot)
    idot = np.reshape(np.repeat(self_dot, ijdot.shape[0]), ijdot.shape)
    jdot = np.reshape(np.repeat(self_dot, ijdot.shape[0]), ijdot.shape).T
    sim_mat = np.triu(ijdot / (idot + jdot - ijdot))
    np.fill_diagonal(sim_mat,0)
    mean_sim = np.mean(sim_mat[np.nonzero(sim_mat)])
    return(mean_sim)

def Sim_loop(data, all_clust, sub_clust):
    sim_list = list(range(len(sub_clust)))
    for i in range(len(sub_clust)):
        curr_clust = data[np.where(all_clust == sub_clust[i])[0]]
        sim_list[i] = Similarity(curr_clust)
    return(sim_list)



def state_stats(fin_clust, all_clust):
    import more_itertools as mit

    p_state, m_dwell = np.zeros(len(fin_clust)),np.zeros(len(fin_clust))
    full_vec = list(range(len(fin_clust)))
    for i in range(len(fin_clust)):
        p_state[i] = len(np.where(all_clust == fin_clust[i])[0])/len(all_clust)

        dur_list = [list(group) for group in mit.consecutive_groups(np.where(all_clust == fin_clust[i])[0])]
        vec = []
        for t in range(len(dur_list)):
            vec = np.append(vec, len(dur_list[t]))
        m_dwell[i] = np.mean(vec)
        full_vec[i] = vec
    return(p_state, m_dwell, full_vec)


def null_states(fin_clust, data):
    import random
    import more_itertools as mit

    all_states = np.arange(1,len(fin_clust)+1)
    rand_states = np.array(random.choices(all_states, k = data.shape[0]))
    dur_list = [list(group) for group in mit.consecutive_groups(rand_states)]
    vec = []
    for t in range(len(dur_list)):
        vec = np.append(vec, len(dur_list[t]))
    null_m_dwell = np.mean(vec)
    return(null_m_dwell)


