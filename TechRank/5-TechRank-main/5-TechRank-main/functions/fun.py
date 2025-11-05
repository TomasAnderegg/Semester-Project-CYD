import pandas as pd
import networkx as nx
import os
import requests
import random
import arrow
import string
import json
import math
import numpy as np
import requests
import urllib.parse

from geopy.geocoders import Nominatim
import geopandas
import geocoder
geolocator = Nominatim(user_agent='myapplication')

import matplotlib.pyplot as plt

from typing import List, Dict
import classes

from pandas import json_normalize 

from networkx.algorithms import bipartite


def df_from_api_CB (query, cb_table):
    """Creates the DataFrame of the data from the CrunchBase (CB) API

    Args:
        - query: CB query that specifies what fields we are needed 
        - cb_table: typology of CB table (organizations or investors or people...)

    Return:
        - df: DataFrame extracted from CB
    """

    api_key = os.getenv("CRUNCHBASE_API_KEY") # extract API key 
    base_url = "https://api.crunchbase.com/api/"
    params = {"user_key" : api_key}

    url_table = "https://api.crunchbase.com/api/v4/searches/" + cb_table

    r = requests.post(url_table, json=query, params=params)

    json_data = json.loads(r.text)

    if 'entities' not in json_data:
        print(json_data)
        return

    df = pd.json_normalize(json_data['entities'])

    return df


def CB_data_cleaning (
    df: pd.DataFrame, 
    to_drop: List[str], 
    to_rename: Dict[str, str], 
    to_check_double: Dict[str, str],
    drop_if_nan: List[str], 
    sort_by: str = ""):
    """Performs the Data Cleaning part of the CB dataset

    Args:
        - df: dataset to clean
        - to_drop: columns to drop
        - to_rename: columns to rename and new name
        - to_check_double: columns to check. If they bring additional value and,
                           in case they don't, drop them
        - sort_by: column by which sort values

    Return:
        - df: cleaned dataset
    """

    df = df.drop(to_drop, axis=1, errors='ignore')
    df = df.rename(columns = to_rename)

    for key, item in to_check_double.items():
        # item does not bring new info:
        if (df[key] == df[item]).all() == True: 
            df = df.drop([item], axis=1)

    # drop if nan
    if len(drop_if_nan)>0:
        for to_drop in drop_if_nan:
            df = df.dropna(subset=[to_drop])

    if len(sort_by)>0:
        df = df.sort_values(sort_by)

    sort_by

    return df


def field_extraction (field_name: str, df: pd.DataFrame):
    """Extract only companies in a specific field according to the words contained in 
    the description
    
    Arg:
        - field_name: name of the field
        
    Return:
        - 
    """

    flag_cybersecurity = False
    
    if field_name == 'cybersecurity':
        field_words = [
                        "cybersecurity"
                        "confidentiality",
                        "integrity",
                        "availability",
                        "secure",
                        "security",
                        "safe",
                        "reliability",
                        "dependability",
                        "confidential",
                        "confidentiality",
                        "integrity",
                        "availability",
                        "defence",
                        "defensive",
                        "privacy"
                        ]
        flag_cybersecurity = True 

    elif field_name == 'medicine':
        field_words = ["cure",
                        'medicine',
                        'surgery',
                        'doctors',
                        'nurses',
                        'hospital',
                        'medication',
                        'prescription',
                        'pill',
                        'health',
                        'cancer',
                        'antibiotic',
                        'HIV',
                        'cancers',
                        'disease',
                        'resonance',
                        'rays',
                        'CAT',
                        'blood',
                        'blood transfusion',
                        'accident',
                        'injuries',
                        'emergency',
                        'poison',
                        'transplant',
                        'biotechnology',
                        'health care',
                        'healthcare',
                        'health-tech',
                        'genetics',
                        'DNA',
                        'RNA',
                        'lab',
                        'heart',
                        'lung',
                        'lungs',
                        'kidneys',
                        'brain',
                        'gynaecologist',
                        'cholesterol',
                        'diabetes',
                        'stroke',
                        'infections',
                        'infection',
                        'ECG',
                        'sonogram'
                        ]
        

    # flag that identifies if we have applied the filter to select only companies in cybersecurity
    df = df.loc[df["short_description"].apply(lambda x: check_desc(x, field_words))]

    return df, flag_cybersecurity


def extract_data_from_column(column, get_what:str):
    """Extracts data from complex columns

    Args:
        - column: data to be changed
        - get_what: value or values we are interested in

    Return:
        - column: cleaned data
    """

    if isinstance(column, pd.core.series.Series):
        column = column.values

    i = 0
    for line in column:
        
        if issubclass(type(line), float) == False:
            column[i] = [ x.get(get_what) for x in line]
        
        i = i+1

    return column


def extract_classes_company_tech(df):
    """Extracts the dictionaries of Companies and Technologies 
    from the dataset and create the network
    
    Args:
        - df: dataset

    Return:
        - dict_companies: dictionary of companies
        - dict_tech: dictionary of technologies
        - B: graph that links companies and technologies 
    """
    
    # dictionary of companies: name company: class Company
    dict_companies = {}
    # dictionary of technologies: name technology: class Technology
    dict_tech = {}
    
    # initialization bipartite graph:
    B = nx.Graph()

    # define Nomiation
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent='myapplication')
    i = 0 # count rows analysed (needed for a trick explained below)

    for index, row in df.iterrows(): # for each company
        
        # location extraction:
        if 'location_comp' in row:
            location_df = row['location_comp']
            location_company = {x.get('location_type'):x.get('value') for x in location_df}
        else:
            location_company = {
                'country_code': row['country_code'], 
                'region': row['region'], 
                'city': row['city']
                }

        i = i + 1

        if i%100==0: # trick to avoid to have an error "GeocoderUnavailable" due to the fact that  Nominatim suggests avoiding extensive use. So, it blocks us after a while
            # in this way we always renew Nominatim every 100 rows.
            del Nominatim
            from geopy.geocoders import Nominatim
            str_name = random.choice(string.ascii_letters)
            geolocator = Nominatim(user_agent=str_name)

        # extraction latitude and longitude:
        if pd.isnull(row['city']) or pd.isnull(row['region']) or pd.isnull(row['country_code']): # one is nan
            location = "Not available"
            lat_c = 90
            lon_c = 180
        else:
            str_place = row['city'] + ', ' + row['region'] #+ ', ' +  row['country_code']
            # add the country code would give problem in getting the url
            
            """location = geolocator.geocode(str_place) # coversion to conventional address (valid for the next command)
            if location is not None: # not null
                lat_c = location.latitude
                lon_c = location.longitude"""

            # url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(str_place) +'?format=json'
            # response = requests.get(url).json()

            # if len(response)>0: # not null
            #     lat_c = response[0]["lat"]
            #     lon_c = response[0]["lon"]
            
            # else: # if None, we set some values very far away
            #     print(f"{str_place} is not a good address")
            #     lat_c = 90
            #     lon_c = 180

            # url = f"https://nominatim.openstreetmap.org/search/{urllib.parse.quote(str_place)}?format=json"
            # headers = {"User-Agent": "TechRankApp/1.0 (contact@epfl.ch)"}  # obligatoire selon Nominatim policy

            # try:
            #     r = requests.get(url, headers=headers, timeout=5)
            #     if r.status_code == 200:
            #         try:
            #             data = r.json()
            #             if len(data) > 0:
            #                 lat_c = float(data[0]["lat"])
            #                 lon_c = float(data[0]["lon"])
            #             else:
            #                 print(f"⚠️ Adresse introuvable: {str_place}")
            #                 lat_c, lon_c = 90, 180
            #         except ValueError:
            #             print(f"⚠️ Réponse non JSON pour {str_place}: {r.text[:100]}")
            #             lat_c, lon_c = 90, 180
            #     else:
            #         print(f"⚠️ Erreur HTTP {r.status_code} pour {str_place}")
            #         lat_c, lon_c = 90, 180
            # except requests.exceptions.RequestException as e:
            #     print(f"⚠️ Erreur réseau pour {str_place}: {e}")
            #     lat_c, lon_c = 90, 180



        # Companies:
        comp_name = row['name']

        c = classes.Company(
            id = row['uuid'],
            name = comp_name,
            location = location_company,
            technologies = row['category_groups'],
            lat = 0.0,#lat_c,
            lon = 0.0#lon_c
                   )

        # if CB rank
        if 'rank_company' in df.columns:
            c.rank_CB = row['rank_company']
        elif 'rank' in df.columns:
            c.rank_CB = row['rank']
        
        dict_companies[comp_name] = c

        B.add_node(comp_name, bipartite=0)
        
        # Technologies:
        if issubclass(type(row['category_groups']), List):
            for tech in row['category_groups']:
                t = classes.Technology(name=tech)
                dict_tech[tech] = t

                B.add_node(tech, bipartite=1)

                # add edges
                B.add_edge(comp_name, tech)
        else:
            t = classes.Technology(name=row['category_groups'])
            dict_tech[tech] = t   

            B.add_node(tech, bipartite=1)

            # add edges
            B.add_edge(comp_name, tech)

    return dict_companies, dict_tech, B


def extract_classes_investors(df):
    """Extracts the dictionaries of Investors from the dataset
    
    Args:
        - df: dataset

    Return:
        - dict_inv: dictionary of Investors
    """
    
    # dictionary of investors: name investors: class Investor
    dict_inv = {}

    geolocator = Nominatim(user_agent='my-agent')
    
    for index, row in df.iterrows():   

        
        # location extraction:
        if 'location_comp' in row:
            location_df = row['location_comp']
            location_company = {x.get('location_type'):x.get('value') for x in location_df}
        else:
            location_company = {
                'country_code': row['country_code'], 
                'region': row['region'], 
                'city': row['city']
                }

        # extraction latitude and longitude:
        #lat, lon  = extract_coordinates_location(location_company)
        lat = 0
        lon = 0
    
        # Companies:
        inv_name = row['name']
        investor_type = row['investor_types']

        i = classes.Investor(
            id = row['uuid'],
            name = inv_name,
            location = location_company,
            investor_type = investor_type
                   )
        
        dict_inv[inv_name] = i

    return dict_inv


def check_desc(line, words):
    """Check if at more than one word in a set of words is included in line (a string)
    """
    
    if isinstance(line, float): # line is not a string
        return False
    
    if sum(w in line.lower() for w in words) > 1:
    #if any(w in line for w in words):
        return True
    
    return False


def extract_class_investor(df):
    """ Extracts the dictionaries of Investors attributes from the dataset 
    
    Args:
        - df: dataset

    Return:
        - dict_inv: dictionary of investors
    """

    # dictionary of investors: name investments: class investors
    dict_investors = {}
    
    for index, row in df.iterrows():   

        location_investor = {
            'country_code': row['country_code'], 
            'region': row['region'], 
            'city': row['city']
            }  

        # extraction latitude and longitude:
        lat, lon  = extract_coordinates_location(location_company)

        # Investor:
        inv_name = row['name'] 

        i = classes.Investor(
            id = row['uuid'],
            name = inv_name,
            type = row['type'],
            location = location_investor,
            lat = lat,
            lon = lon,
            investor_type = row['investor_types'],
            investment_count = row['investment_count']
            )
        
        dict_investors[inv_name] = i

    return dict_investors
    

def nx_dip_graph_from_pandas(df):
    """ Creates the bipartite graph from the dataset

    bipartite = 0 => company
    bipartite = 1 => other value

    Args:
        - df: Datafame

    Return:
        - B: bipartite graph 
    """

    df_columns = list(df.columns)
    df_columns = df_columns[0]

    B = nx.Graph()
    
    for name, value in df.iterrows():
    
        value = value[df_columns]
        
        if issubclass(type(value), float) == False:
        
            B.add_node(name, bipartite=0)

            if issubclass(type(value), str): # if only one value
                B.add_node(value, bipartite=1)
                B.add_edge(name, value)

            else: # if value has more values (list)
                for x in value:
                    B.add_node(x, bipartite=1)
                    B.add_edge(name, x)

    return B


def extract_nodes(G, bipartite_set) -> List:
    """Extract nodes from the nodes of one of the bipartite sets

    Args:
        - G: graph
        - bipartite_set: select one of the two sets (0 or 1)

    Return:
        - nodes: list of nodes of that set
    """

    nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == bipartite_set]

    return nodes


def filter_dict(G, percentage, set1, set2):
    """Selects values to delete from the graph G according to ...

    Args:
        - G: graph 
        - percentage: percentage of entities to keep
        - set1: first group of nodes
        - set2: second group of nodes

    Return:
        - to_delete: list of values to delete
    """

    degree_set2 = list(dict(nx.degree(G, set2)).values())
    
    threshold_companies = math.ceil( len(set2)/percentage )
    
    
    if threshold_companies > np.max(degree_set2): # not okay because we would not plot anything
        threshold_companies=np.mean(degree_set2)
    
    dict_nodes = nx.degree(G, set1) 
    
    to_delete= []
    
    # Iterate over all the items in dictionary
    for (key, value) in dict(dict_nodes).items():
        
        if value <= threshold_companies:
            to_delete.append(key)
    
    return to_delete


def plot_bipartite_graph(G, small_degree=True, percentage=10, circular=False):
    """ Plots the bipartite network ...

    Args:
        - G: graph 
        - small_degree: if plot the nodes with small degree
            if True: plot all
            if False: delete nodes with low degree
        - percentage
        - circular
    """

    set1 = [node for node in G.nodes() if G.nodes[node]['bipartite']==0]
    set2 = [node for node in G.nodes() if G.nodes[node]['bipartite']==1]

    if circular == False:
        pos = nx.spring_layout(G) # positions for all nodes
    else:
        pos = nx.circular_layout(G)

    if small_degree == False: # don't plot nodes with low number of edges
        to_delete = filter_dict(G, percentage, set1, set2)

        G_filtered = G.copy()
        G_filtered.remove_nodes_from(to_delete)
        G_filtered.remove_nodes_from(list(nx.isolates(G_filtered)))

        pos = plot_bipartite_graph(G_filtered, small_degree=True, percentage=percentage, circular=circular)
        
        return pos

    if len(set1)>=20:
        plt.figure(1,figsize=(25,15))
    else: 
        plt.figure(1,figsize=(19,13)) 
    
    plt.axis('off')


    # company, value = bipartite.sets(G)
    # subsituted with:
    company = [node for node in G.nodes() if G.nodes[node]['bipartite']==0]
    value = [node for node in G.nodes() if G.nodes[node]['bipartite']==1]

    # calculate degree centrality
    companyDegree = nx.degree(G, company) 
    valueDegree = nx.degree(G, value)


    # nodes
    nx.draw_networkx_nodes(G,
                           pos,
                           nodelist=company,
                           node_color='r',
                           node_size=[v * 100 for v in dict(companyDegree).values()],
                           alpha=0.25,
                           label=company)

    nx.draw_networkx_nodes(G,
                           pos,
                           nodelist=value,
                           node_color='b',
                           node_size=[v * 200 for v in dict(valueDegree).values()],
                           alpha=0.25,
                           label=value)

    nx.draw_networkx_labels(G, pos, {n: n for n in company}, font_size=15)
    nx.draw_networkx_labels(G, pos, {n: n for n in value}, font_size=15)

    # edges
    nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.4)


    plt.axis('off')
    axis = plt.gca()
    axis.set_xlim([1.2*x for x in axis.get_xlim()])
    axis.set_ylim([1.2*y for y in axis.get_ylim()])
    plt.tight_layout()

    return pos


def degree_bip(G):
    """ Gives the degree of each node of the two bipartite graphs.

    degree: number of links

    Args:
        - G: graph

    Return:
        - degree_set0(1): degree of the bipartite set 0(1)
    """

    set0 = extract_nodes(G, 0)
    set1 = extract_nodes(G, 1)

    # calculate degree centrality
    degree_set0 = nx.degree(G, set0) 
    degree_set1 = nx.degree(G, set1)

    return dict(degree_set0), dict(degree_set1)


def insert_data_classes(dict_class: Dict, new_data: Dict, feature: str):
    """ 

    Arg:
        - dict_class: (name, class)
        - new_data: dict or DegreeView with the name and the value of the feature
        - feature: name of the feature in class to update 

    Return:
        - dict_class: (name, class) with updated class
    """
    
    for name, value in new_data.items():
        
        if name not in dict_class.keys():
            print(f"Error: try to add a feature for {name}, which is not in class")
            return
        
        c = dict_class[name]
        
        setattr(c, feature, value)
    
    return dict_class

