import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def M_test_triangular(M, flag_cybersecurity=False):
    """Test the triangularity of M matrix"""

    user_edits_sum = M.sum(axis=1).flatten()
    article_edits_sum = M.sum(axis=0).flatten()

    user_edits_order = user_edits_sum.argsort()
    article_edits_order = article_edits_sum.argsort()

    M_sorted = M[user_edits_order,:]

    if len(M_sorted.shape)>2: # the matrix in inside the first
        M_sorted = M_sorted[0] # so it becomes of size 2

    M_sorted_transpose = M_sorted.transpose()

    M_sorted_transpose = M_sorted_transpose[article_edits_order,:]

    if len(M_sorted_transpose.shape)>2: # the matrix in inside the first
        M_sorted_transpose = M_sorted_transpose[0] # so it becomes of size 2

    M_sorted_sorted = M_sorted_transpose#.transpose()

    params = {
        'axes.labelsize': 18,
        'axes.titlesize':28, 
        'legend.fontsize': 22, 
        'xtick.labelsize': 16, 
        'ytick.labelsize': 16}

    plt.figure(figsize=(10, 10))
    plt.rcParams.update(params)
    plt.imshow(M_sorted_sorted, cmap=plt.cm.bone, interpolation='nearest')
    plt.xlabel("Companies")
    plt.ylabel("Technologies")

    if flag_cybersecurity==False: # all fields
        name_plot_M = f'plots/M_triangulize/matrix_{str(M_sorted.shape)}'
    else: # only companies in cybersecurity
        name_plot_M = f'plots/M_triangulize/cybersec_matrix_{str(M_sorted.shape)}'

    plt.savefig(f'{name_plot_M}.pdf')
    plt.savefig(f'{name_plot_M}.png')
    plt.show()

    return


def zero_order_score(M):
    
    # k_c = list(company_degree.values()) # companies
    # k_t = list(tech_degree.values()) # technologies
    
    k_c = M.sum(axis=1)
    k_t = M.sum(axis=0)


    return k_c, k_t


def Gct_beta(M, c, t, k_c, beta):

    num = (M[c,t]) * (k_c[c] ** (- beta))

    # sum over the technologies
    M_t = M[:,t].flatten()
    k_c_beta = [x ** (-1 * beta) for x in k_c]

    den = float(np.dot(M_t, k_c_beta))
    
    return num/den


def Gtc_alpha(M, c, t, k_t, alpha):
    
    num = (M.T[t,c]) * (k_t[t] ** (- alpha))
    
    # sum over the companies
    M_c = M[c,:].flatten()
    k_t_alpha = [x ** (-1 * alpha) for x in k_t]
    
    type(M_c)
    type(k_t_alpha)
    
    den = float(np.dot(M_c, k_t_alpha))
    
    return num/den


def make_G_hat(M, alpha=1, beta=1):
    '''G hat is Markov chain of length 2
    Gct is a matrix to go from  companies to technologies and  
    Gtc is a matrix to go from technologies to companies'''
    
    # zero order score
    k_c, k_t = zero_order_score(M)
    
    # allocate space
    G_tc = np.zeros(shape=M.T.shape)
    G_ct = np.zeros(shape=M.shape)
    
    # Gct_beta
    for [c, t], val in np.ndenumerate(M):
        G_ct[c,t] = Gct_beta(M, c, t, k_c, beta)
    
    # Gtc_alpha
    for [t, c], val in np.ndenumerate(M.T):
        G_tc[t,c] = Gtc_alpha(M, c, t, k_t, alpha)
    
    return {'G_ct': G_ct, "G_tc" : G_tc}


def next_order_score(G_ct, G_tc, fitness_prev, ubiquity_prev):
    '''Generates w^(n+1) from w^n
    '''
    
    fitness_next = np.sum( G_ct * ubiquity_prev, axis=1 )
    ubiquity_next = np.sum( G_tc * fitness_prev, axis=1 )
    
    return fitness_next, ubiquity_next


def generator_order_w(M, alpha, beta):
    """Generates w_t^{n+1} and w_c^{n+1}
    
    fitness_next = w_t next
    ubliq_next = w_c next
    
    """
    
    # transition probabilities
    G_hat = make_G_hat(M, alpha, beta)
    G_ct = G_hat['G_ct']
    G_tc = G_hat['G_tc']
    
    # strating point
    fitness_0, ubiquity_0  = zero_order_score(M)
    
    fitness_next = fitness_0
    ubiquity_next = ubiquity_0
    i = 0
    
    while True:
        
        fitness_prev = fitness_next
        ubiquity_prev = ubiquity_next
        i += 1
        
        fitness_next, ubiquity_next = next_order_score(G_ct, G_tc, fitness_prev, ubiquity_prev)
        
        yield {'iteration':i, 'fitness': fitness_next, 'ubiquity': ubiquity_next}
        

def w_stream(M, i, alpha, beta):
    """get a specific itelration of w, 
    but in a memory safe way so we can calculate many generations"""
    
    if i < 0:
        raise ValueError
        
    for j in generator_order_w(M, alpha, beta):
        if j[0] == i:
            return {'fitness': j[1], 'ubiquity': j[2]}
            break


def find_convergence(M, 
                    alpha, 
                    beta, 
                    fit_or_ubiq, 
                    do_plot=False, 
                    flag_cybersecurity=False,
                    preferences = ''):
    '''TechRank evolution: finds the convergence point (or gives up after 1000 iterations)
    
    Args:
        - 
        
    Return: 
        - 
        
    '''
    
    # technologies or company
    if fit_or_ubiq == 'fitness':
        M_shape = M.shape[0]
        name = 'Companies'
    elif fit_or_ubiq == 'ubiquity':
        name = 'Technologies'
        M_shape = M.shape[1]

    #print(name)
    
    rankings = list()
    scores = list()
    
    prev_rankdata = np.zeros(M_shape)
    iteration = 0
    

    weights = generator_order_w(M, alpha, beta)


    stops_flag = 0

    for stream_data in weights:
        
        iteration = stream_data['iteration']
        
        data = stream_data[fit_or_ubiq] # weights
        
        rankdata = data.argsort().argsort()

        # print(f"iteration : {iteration}")
        
        if iteration==1:
            # print(iteration, rankdata)
            initial_conf = rankdata

        # print(f"Iteration: {iteration} stops flag: {stops_flag}")

        # stops in case algorithm does not change for some iterations
        if stops_flag==10:
            print(f"Converge at {iteration}")
            convergence_iteration = iteration
            for i in range(90):
                rankings.append(rankdata)
                scores.append(data)
            break

        # test for convergence, in case break
        elif np.equal(rankdata,prev_rankdata).all(): # no changes
            if stops_flag==0:
                convergence_iteration = iteration
            stops_flag += 1

            # reappend two times to make plot flat
            rankings.append(rankdata)
            scores.append(data)
            prev_rankdata = rankdata

        # max limit
        elif iteration == 5000: 
            print("We break becuase we reach a too high number of iterations")
            convergence_iteration = iteration
            break

        # go ahead
        else: 
            rankings.append(rankdata)
            scores.append(data)
            prev_rankdata = rankdata
            stops_flag = 0

            
    # print(iteration, rankdata)
    final_conf = rankdata
    
    # plot:
    if do_plot and iteration>2:

        params = {
            'axes.labelsize': 26,
            'axes.titlesize':28, 
            'legend.fontsize': 22, 
            'xtick.labelsize': 16, 
            'ytick.labelsize': 16}

        plt.figure(figsize=(10, 10))
        plt.rcParams.update(params)
        plt.xlabel('Iterations')
        plt.ylabel('Rank, higher is better')
        plt.title(f'{name} rank evolution')
        plt.semilogx(range(1,len(rankings)+1), rankings, '-,', alpha=0.5)

        # save figure 
        if flag_cybersecurity==False:
            name_plot = f'plots/rank_evolution/techrank_{name}_{str(M_shape)}_{str(preferences)}'
        else:
            name_plot = f'plots/rank_evolution/techrank_cybersecurity_{name}_{str(M_shape)}_{str(preferences)}'
        plt.savefig(f'{name_plot}.pdf')
        plt.savefig(f'{name_plot}.png')

        
    return {fit_or_ubiq: scores[-1], 
            'iteration': convergence_iteration, 
            'initial_conf': initial_conf, 
            'final_conf': final_conf}


def rank_df_class(convergence, dict_class):
    """Creates a Dataframe to have a representation of the evolution with the iterations and the rank and 
    update the class with the rank found with the find_convergence algorithm
    """
    
    if 'fitness' in convergence.keys():
        fit_or_ubiq = 'fitness'
    elif 'ubiquity' in convergence.keys():
        fit_or_ubiq = 'ubiquity'
    
    list_names = [*dict_class]
    
    n = len(list_names)

    if hasattr(list_names[0], 'rank_CB'):
        columns_final = ['initial_position', 'final_configuration', 'degree', 'techrank', 'rank_CB']
    else:
        columns_final = ['initial_position', 'final_configuration', 'degree', 'techrank']

    df_final = pd.DataFrame(columns=columns_final, index=range(n))

    # print(f"----{n}----{len(convergence['initial_conf'])}----")
    if n>len(convergence['initial_conf']):
        n = n-1
    
    for i in range(n):
        # print(i)
        
        name = list_names[i]
        
        ini_pos = convergence['initial_conf'][i] # initial position
        final_pos = convergence['final_conf'][i] # final position
        rank = round(convergence[fit_or_ubiq][i], 3) # final rank rounded
        degree = dict_class[name].degree
        
        df_final.loc[final_pos, 'final_configuration'] = name
        df_final.loc[final_pos, 'degree'] = degree
        df_final.loc[final_pos, 'initial_position'] = ini_pos
        df_final.loc[final_pos, 'techrank'] = rank


        if hasattr(dict_class[name], 'rank_CB'):
            rank_CB = dict_class[name].rank_CB
            df_final.loc[final_pos, 'rank_CB'] = rank_CB
        
        
        # update class's instances with rank
        dict_class[name].rank_algo = rank
    
    
    return df_final, dict_class


def w_star_analytic (M, alpha, beta, ua, dict_class=[]):
    """Find the weights evolution analytically

    Args:
        -
        - ua: defines if we are working with 'Technologies' or 'Companies'
        - dict_class: dictionary (of companies or technologies) to update (only the rank_analytic attribute).
                      In case we don't pass it is zero

    w^{*}_c (alpha,beta) = (\sum_{t=1}^{N_t} M_{ct}k_{t}^{-alpha})k_{c}^{-beta} 
    w^{*}_t (alpha,beta) = (\sum_{c=1}^{N_c}  M_{ea}k_{c}^{-beta})k_{t}^{-alpha}
    """
    
    k_c = M.sum(axis=1) # aka k_c summing over the rows
    k_p = M.sum(axis=0) # aka k_p summering over the columns
    
    A = 1
    B = 1
    
    def Gcp_denominator(M, p, k_c, beta):
        M_p = M[:,p]
        k_c_beta = k_c ** (-1 * beta)
        return np.dot(M_p, k_c_beta)
    
    def Gpc_denominator(M, c, k_p, alpha):
        M_c = M[c,:]
        k_p_alpha = k_p ** (-1 * alpha)
        return np.dot(M_c, k_p_alpha)
    
    if ua == 'Companies': 
        w_star = np.zeros(shape=M.shape[0])

        for c in range(M.shape[0]):
            summand = Gpc_denominator(M, c, k_p, alpha)
            k_beta = (k_c[c] ** (-1 * beta))
            w_star[c] = A * summand * k_beta
    
    elif ua == 'Technologies':
        w_star = np.zeros(shape=M.shape[1])
    
        for p in range(M.shape[1]):
            summand = Gcp_denominator(M, p, k_c, beta)
            k_alpha = (k_p[p] ** (-1 * alpha))
            w_star[p] = B * summand * k_alpha
  
    w_star_dict = {}

    if len(dict_class)>0: # dict class has to be updated
        # Update the rank_analytic attribute
        list_names = [*dict_class]
        n = len(list_names)

        #print(f"n: {n} -- w_star: {len(w_star)}")
        if n>len(w_star):
            n = len(w_star)

        for i in range(n):
            name = list_names[i]
            dict_class[name].rank_analytic = w_star[i]
            w_star_dict[name] = w_star[i] # add element to the dict
        
    return w_star_dict, dict_class

