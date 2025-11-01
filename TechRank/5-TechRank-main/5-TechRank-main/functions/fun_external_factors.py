import os
import requests
import arrow
import json
import math
import geopandas

import haversine as hs
import pandas as pd
import seaborn as sns
import networkx as nx
import scipy.stats as ss
import haversine as hs
import matplotlib.pyplot as plt

import numpy as np
import geopy.geocoders as geocoders

from geopy.geocoders import Nominatim
from typing import Dict
from importlib import reload

import functions.fun_meth_reflections
from functions.fun_meth_reflections import w_star_analytic

geolocator = Nominatim(user_agent='myapplication')


def rank_comparison(a_ranks_sorted: Dict[str, float], 
                    b_ranks_sorted: Dict[str, float], 
                    do_plot=False):
    """ Returns the Spearman correlation of two ranks (in a Dict form) already sorted
    
    Args:
        - a_ranks_sorted: first rank to compare
        - b_ranks_sorted: second rank to compare

    Return:
        - sper_corr: Spearman correlation
    """
    
    a_list = list()
    b_list = list()
    
    for a_name, a_rank in a_ranks_sorted.items():
        
        #find this element in list b
        for b_name, b_rank in b_ranks_sorted.items():
            
            if a_name == b_name: # if same element in a and b
                a_list.append(a_rank)
                b_list.append(b_rank)
                break
    
    # after the loop we have wo lists, a_list and b_list, for which, in each position, we have the rank of the same element. 
    # This is needed to find the correlation
    
    # plot        
    if do_plot:    
        plt.figure(figsize=(10,20))
        plot([1,2], [a_list, b_list], '-o')
        plt.show()

    sper_corr = ss.spearmanr(a_list, b_list)

    return sper_corr


def calibrate_analytic(M, 
                       ua, 
                       dict_class, 
                       exogenous_rank, 
                       index_function, 
                       title, 
                       do_plot=False,
                       flag_cybersecurity = False,
                       preferences = ''):
    """ Returns the top parameters after the greed search
    
    Args:
        - M: triangular matrix
        - ua: defines if we are working with 'Technologies' or 'Companies'
        - dict_class: dictionary of technologies of companies according to ua
        - exogenous_rank: basetruth
        - groundtruth_dict: dict to use as groundtruth
        - index_function: describe how mapping the area for the greed search
        - title: title of the plot (in case we have one)
        - do_plot: indicate if plotting
        - flag_cybersecurity: identifies if we are in the cybersecurity field only
        
        
    Return:
        - top_spearman: dict including 
                                'spearman': ..
                                'alpha': best alpha
                                'beta': best beta
                                'ua'
    """
    
    # ua
    if ua == 'Companies':
        w_star_type = 'w_star_c'
    elif ua == 'Technologies':
        w_star_type = 'w_star_p'
    
    # Create the map:
    squarelen = np.arange(100) # range in which iterate

    # map the position in the grid with the parameter value
    alpha_range = [index_function(x) for x in squarelen]
    beta_range = [index_function(x) for x in squarelen]
    
    landscape = np.zeros(shape=(len(list(alpha_range)),len(list(beta_range))))

    # Structure of the result
    top_spearman = {'spearman':None,'alpha':None, 'beta':None, 'ua':ua}

    # Grid search among all alphas and all betas
    for alpha_index, alpha in enumerate(alpha_range): #_index is the count of the current iteration
        for beta_index, beta in enumerate(beta_range):
            
            # find the analytic weights with that specific alpha and beta:
            [w_rank, dict] = w_star_analytic(M, alpha, beta, ua, dict_class) # it already returns a dictionary
            
            # w_rank must be normalized to mach the exogenous rank structure
            max_w_rank = max(w_rank.values())
            w_rank = {name: inv/max_w_rank for (name, inv) in w_rank.items()}

            
            # Spearman correlation between the created rank and the benchmark
            spearman = rank_comparison(w_rank, exogenous_rank)
            
            # the spearman variable contains both the correlation and the pvalue:
            sper_corr = spearman[0]
            sper_pvalue = spearman[1]

            if sper_pvalue < 1: # statistically significant
                landscape[alpha_index][beta_index] = sper_corr
                
                if (not top_spearman['spearman']) or (sper_corr > top_spearman['spearman']):
                    top_spearman['spearman'] = sper_corr
                    top_spearman['alpha'] = alpha
                    top_spearman['beta'] = beta
            else:
                landscape[alpha_index][beta_index] = np.nan

    if np.isnan(landscape).all(): # for all cases teh pvalue was too high 
        print("No significant results")
        return 

    # plot
    if do_plot:

        params = {
            'axes.labelsize': 26,
            'axes.titlesize':26, 
            'legend.fontsize': 19, 
            'xtick.labelsize': 18, 
            'ytick.labelsize': 18}

        fig,ax=plt.subplots(1,1, figsize=(10, 10))
        plt.rcParams.update(params)
        heatmap = plt.imshow(landscape, interpolation='nearest', vmin=-1, vmax=1)
        heatmap = plt.pcolor(landscape)
        colorbar = plt.colorbar(heatmap)
        plt.locator_params(nbins=2, axis='x')
        plt.xlabel(r'$ \beta $')
        plt.xticks(squarelen, beta_range, rotation=90)
        plt.ylabel(r'$ \alpha $')
        plt.yticks(squarelen, alpha_range)

        # reduce the density of ticks
        for i, tick in enumerate(ax.yaxis.get_ticklabels()):
            if i % 4 != 0:
                tick.set_visible(False)
        for i, tick in enumerate(ax.xaxis.get_ticklabels()):
            if i % 4 != 0:
                tick.set_visible(False)

        plt.title(title)

        # save
        if flag_cybersecurity == False:
            name_plot = f'plots/parameters_optimization/par_optim_{ua}_{len(dict_class)}_{str(preferences)}'
        else:
            name_plot = f'plots/parameters_optimization/par_optim_cybersecurity_{ua}_{len(dict_class)}_{str(preferences)}'

        plt.savefig(f'{name_plot}.pdf')
        plt.savefig(f'{name_plot}.png')

    return top_spearman

    
def create_exogenous_rank(ua, dict_class, preferences: Dict[str, float]):
    """Create the exogenous rank used as groundtruth for the model calibration. It accounts for the investors' preferences 
    including exogenous factors, such as previous investments, geographical distance ecc.

    Arg:
        - ua: defines if we are working with 'Technologies' or 'Companies'
        - dict_class: dictionary of technologies of companies according to ua
        - preferences: investors' preferences expressed as Dict(name:percentage). The sum must be 100.
    Return:
        - exogenous_rank
    """

    if sum(preferences.values())!=100: # the sum of all the preferences is not 100
        print("Error: the investors' preferences percentage must sum to 100%")
        return

    # initialize resulting dict
    exogenous_rank = {name:0 for name in dict_class.keys()}

    # check which factors are included in the preferences
    # work on them if they are included and their preference percentage is >0

    # 1: amount previous investments -------------------------------------------------------------
    if "previous_investments" in preferences.keys() and preferences["previous_investments"]>0:
       
        # dictionary name company: total_prev_inv
        dict_comp_inv = {name: row.tot_previous_investments for (name, row) in dict_class.items()}

        # normalize:
        max_inv = max(dict_comp_inv.values())

        if max_inv == 0: 
            print('Probably you have forgotten to first insert the data about previous investments')
            return 

        dict_comp_inv_norm = {name: inv/max_inv for (name, inv) in dict_comp_inv.items()}
 
        # update exogenous_rank
        perc_contribution = preferences["previous_investments"]/100 # percentage of contribution 

        for key, value in exogenous_rank.items():
            exogenous_rank[key] = value + perc_contribution * dict_comp_inv_norm[key]
   
    
    # 2: Crunchbase -------------------------------------------------------------
    if "crunchbase_rank" in preferences.keys() and preferences["crunchbase_rank"]>0:
        if ua=='Technologies': # this factors makes sense only for companies
            print("The CB rank factor can be applied only for companies, not technologies")
            return 
        
        # dictionary name company: CB rank       
        dict_comp_cb = {name: row.rank_CB for (name, row) in dict_class.items()}
        # normalize:
        max_rank = max(dict_comp_cb.values())
        dict_comp_cb_norm = {name: inv/max_rank for (name, inv) in dict_comp_cb.items()}

        # update exogenous_rank
        perc_contribution = preferences["previous_investments"]/100 # percentage of contribution 

        for key, value in exogenous_rank.items():
            exogenous_rank[key] = value + perc_contribution * dict_comp_cb_norm[key]
    
    
    # 3: Geographical position --------------------------------------------------------------------
    if "geo_position" in preferences.keys() and preferences["geo_position"]>0:

        # position of the investor (let us suppose he is in NY)
        city_inv = "London"
        region_inv = "England"
        country_inv = "UK"
        print(f"Investors in {city_inv}")
        str_place = city_inv + ', ' + region_inv + ', ' + country_inv
        location = geolocator.geocode(str_place) # coversion to conventional address (valid for the next command)

        lat_inv = location.latitude
        lon_in= location.longitude
        
        dict_h = {} # dictionary company Haversine distance
        
        for (name, row) in dict_class.items():
            # lat and lon company
            lat_c = row.lat
            lon_c = row.lon

            # Haversine distance between the company and the investor
            h = haversine_distance(lat_c, lon_c, lat_inv, lon_in)

            dict_h[name] = h

        max_h = max(dict_h.values())

        dict_h_norm = {}

        for name, h in dict_h.items():

            x = 1 - h/max_h 

            if x == 0:
                dict_h_norm[name] = 1 
            else:
                dict_h_norm[name] = 1/x 
 
        # update exogenous_rank
        perc_contribution = preferences["geo_position"]/100 # percentage of contribution 

        for key, value in exogenous_rank.items():
            exogenous_rank[key] = value + perc_contribution * dict_h_norm[key]
   
    
    

    return exogenous_rank


def extract_coordinates_location(location_company):
    """Extracts the address, the latitude and the longitude from the dict of the address

    Arg:
        - location_company: location company expressed in words
    Return:
        - lat, lon: position of the location_company
    """
    
    str_place = location_company['city'] + ', ' + location_company['region'] #+ ', ' +  row['country_code']

    address = geolocator.geocode(str_place) # coversion to conventional address (valid for the next command)
    
    lat = address.latitude
    lon = address.longitude
    
    return lat, lon 


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the haversine distance (in km) between two points on heart

    Arg:
        - location_company: location company expressed in words
    Return:
        - lat, lon: position of the location_company

    """

    loc1=(lat1, lon1)
    loc2=(lat2, lon2)
    distance = hs.haversine(loc1,loc2)

    return distance # in km
