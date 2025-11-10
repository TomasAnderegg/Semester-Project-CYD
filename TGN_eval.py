import numpy as np
from utils.data_processing import get_data
from utils import *  # selon ce que tu utilises

DATASET_NAME = "crunchbase"  # à adapter

# 🔹 Chargement des CSV et features .npy
print("🔹 Chargement des CSV et features .npy")

# Déballage de la sortie de get_data()
timestamps, features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = get_data(DATASET_NAME)

# Vérification
print(timestamps.shape)  # devrait correspondre à la taille des timestamps
print(features.shape)    # devrait correspondre à la taille des features

# Informations sur le dataset
print(f"The dataset has {full_data.n_interactions} interactions, involving {full_data.n_nodes} different nodes")
print(f"The training dataset has {train_data.n_interactions} interactions, involving {train_data.n_nodes} different nodes")
print(f"The validation dataset has {val_data.n_interactions} interactions, involving {val_data.n_nodes} different nodes")
print(f"The test dataset has {test_data.n_interactions} interactions, involving {test_data.n_nodes} different nodes")
print(f"The new node validation dataset has {new_node_val_data.n_interactions} interactions, involving {new_node_val_data.n_nodes} different nodes")
print(f"The new node test dataset has {new_node_test_data.n_interactions} interactions, involving {new_node_test_data.n_nodes} different nodes")
print(f"{len(new_node_test_data.node_set)} nodes were used for the inductive testing, i.e. are never seen during training")
