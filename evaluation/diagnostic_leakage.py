import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# ==============================
# DIAGNOSTIC DU DATA LEAKAGE
# ==============================

GRAPH_PATH = "savings/bipartite_invest_comp/networks/bipartite_graph_10000.gpickle"
DATA_CSV = "./data/crunchbase_filtered.csv"

def diagnostic_data_leakage():
    """
    Identifie pourquoi RF obtient AUC=1.0 (trop facile).
    """
    print("\n" + "="*70)
    print("DIAGNOSTIC DU DATA LEAKAGE")
    print("="*70)
    
    # Charger le graphe
    with open(GRAPH_PATH, "rb") as f:
        B = pickle.load(f)
    
    # Charger les données TGN
    df = pd.read_csv(DATA_CSV)
    val_time, test_time = list(np.quantile(df.ts, [0.70, 0.85]))
    
    print(f"\nTimestamps splits:")
    print(f"  Train: ts <= {val_time}")
    print(f"  Val:   {val_time} < ts <= {test_time}")
    print(f"  Test:  ts > {test_time}")
    
    # ============================================================
    # PROBLÈME 1 : Les négatives n'existent JAMAIS dans le graphe
    # ============================================================
    print("\n" + "-"*70)
    print("PROBLÈME 1 : Négatives vs Positives")
    print("-"*70)
    
    # Extraire les positives de test
    test_edges_pos = []
    for u, v, data in B.edges(data=True):
        ts_list = []
        for fr in data.get("funding_rounds", []):
            ann = fr.get("announced_on", None)
            if ann:
                try:
                    ts_list.append(pd.to_datetime(ann).timestamp())
                except:
                    continue
        if not ts_list:
            continue
        first_ts = min(ts_list)
        if first_ts > test_time:
            test_edges_pos.append((u, v, data))
    
    print(f"\nTest positives: {len(test_edges_pos)}")
    
    # Analyser les features des positives
    pos_raised = []
    pos_rounds = []
    for u, v, data in test_edges_pos:
        pos_raised.append(data.get('total_raised_amount_usd', 0))
        pos_rounds.append(data.get('num_funding_rounds', 0))
    
    print(f"\nPOSITIVES (edges qui EXISTENT dans le graphe):")
    print(f"  total_raised_amount_usd:")
    print(f"    - Mean: ${np.mean(pos_raised):,.0f}")
    print(f"    - Median: ${np.median(pos_raised):,.0f}")
    print(f"    - Min: ${np.min(pos_raised):,.0f}")
    print(f"    - Max: ${np.max(pos_raised):,.0f}")
    print(f"    - Non-zero: {(np.array(pos_raised) > 0).sum()}/{len(pos_raised)}")
    print(f"  num_funding_rounds:")
    print(f"    - Mean: {np.mean(pos_rounds):.2f}")
    print(f"    - Non-zero: {(np.array(pos_rounds) > 0).sum()}/{len(pos_rounds)}")
    
    print(f"\n🔴 NÉGATIVES (edges qui N'EXISTENT PAS dans le graphe):")
    print(f"  total_raised_amount_usd = 0 (TOUJOURS)")
    print(f"  num_funding_rounds = 0 (TOUJOURS)")
    
    print("\n[WARNING]  CONCLUSION PROBLÈME 1:")
    print("  Les négatives ont TOUJOURS raised=0 et rounds=0")
    print("  Les positives ont PRESQUE TOUJOURS raised>0 et rounds>0")
    print("  → Le RF peut distinguer avec 100% de précision juste avec ces features!")
    
    # ============================================================
    # PROBLÈME 2 : Les degrés sont aussi un signal parfait
    # ============================================================
    print("\n" + "-"*70)
    print("PROBLÈME 2 : Degrés comme signal")
    print("-"*70)
    
    nodes_comp = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    nodes_inv = [n for n, d in B.nodes(data=True) if d["bipartite"] == 1]
    
    # Calculer les degrés
    comp_degrees = {n: B.degree(n) for n in nodes_comp}
    inv_degrees = {n: B.degree(n) for n in nodes_inv}
    
    print(f"\nDegrés des nœuds:")
    print(f"  Companies - Mean degree: {np.mean(list(comp_degrees.values())):.2f}")
    print(f"  Companies - Nodes with degree=0: {sum(1 for d in comp_degrees.values() if d == 0)}")
    print(f"  Investors - Mean degree: {np.mean(list(inv_degrees.values())):.2f}")
    print(f"  Investors - Nodes with degree=0: {sum(1 for d in inv_degrees.values() if d == 0)}")
    
    # Simuler des négatives
    from utils.utils import RandEdgeSampler
    train_edges = [(u, v) for u, v, data in B.edges(data=True)]
    train_sources = [u for u, v in train_edges]
    train_dests = [v for u, v in train_edges]
    
    sampler = RandEdgeSampler(train_sources, train_dests)
    _, neg_dests = sampler.sample(100)
    
    # Vérifier si les négatives ont des degrés différents
    existing_edges = set(B.edges())
    neg_pairs = [(nodes_comp[i % len(nodes_comp)], neg_dests[i]) for i in range(100)]
    
    neg_exist_in_graph = sum(1 for u, v in neg_pairs if (u, v) in existing_edges)
    
    print(f"\n🔴 Sur 100 négatives générées:")
    print(f"  Existent déjà dans le graphe: {neg_exist_in_graph}")
    print(f"  N'existent pas dans le graphe: {100 - neg_exist_in_graph}")
    
    print("\n[WARNING]  CONCLUSION PROBLÈME 2:")
    if neg_exist_in_graph > 0:
        print("  Certaines 'négatives' existent en fait dans le graphe!")
        print("  → Mais elles ont raised=0 car on les sample APRÈS l'événement")
    else:
        print("  Les négatives n'existent vraiment pas dans le graphe")
        print("  → Elles ont forcément raised=0 et rounds=0")
    
    # ============================================================
    # PROBLÈME 3 : Temporal leakage dans extract_features
    # ============================================================
    print("\n" + "-"*70)
    print("PROBLÈME 3 : Temporal Leakage potentiel")
    print("-"*70)
    
    print("\nDans extract_features(), on utilise edge_ts comme cutoff.")
    print("Mais pour les NÉGATIVES, edge_ts = 0 (dummy timestamp)!")
    print("\nVoyons ce que ça donne:")
    
    # Simuler une négative
    u_neg, v_neg = nodes_comp[0], nodes_inv[0]
    edge_data_neg = B.get_edge_data(u_neg, v_neg)
    
    print(f"\n  Paire négative: ({u_neg}, {v_neg})")
    print(f"  Edge existe dans B? {edge_data_neg is not None}")
    if edge_data_neg:
        print(f"  → raised={edge_data_neg.get('total_raised_amount_usd', 0)}")
        print(f"  → rounds={edge_data_neg.get('num_funding_rounds', 0)}")
        print(f"  [WARNING]  Cette 'négative' a en fait des données!")
    else:
        print(f"  → Edge n'existe pas, donc raised=0, rounds=0")
    
    print("\n[WARNING]  CONCLUSION PROBLÈME 3:")
    print("  Les négatives ont edge_ts=0, donc on ne compte AUCUN funding round")
    print("  → Elles ont TOUJOURS raised=0 et rounds=0")
    print("  → Signal parfait pour les distinguer des positives!")
    
    # ============================================================
    # RÉSUMÉ
    # ============================================================
    print("\n" + "="*70)
    print("RÉSUMÉ DES PROBLÈMES")
    print("="*70)
    print("\n🔴 POURQUOI RF OBTIENT AUC=1.0:")
    print("\n1. POSITIVES (edges réels):")
    print("   - Ont des funding rounds → raised > 0, num_rounds > 0")
    print("   - Features: [u_deg, v_deg, log(raised), rounds]")
    print("   - Exemple: [5, 3, 15.4, 2]")
    
    print("\n2. NÉGATIVES (edges fictifs):")
    print("   - N'ont JAMAIS de funding rounds → raised = 0, num_rounds = 0")
    print("   - Features: [u_deg, v_deg, 0.0, 0]")
    print("   - Exemple: [5, 3, 0.0, 0]")
    
    print("\n3. LE RF APPREND:")
    print("   if (raised == 0 and num_rounds == 0): predict 0 (négative)")
    print("   else: predict 1 (positive)")
    print("   → Précision parfaite!")
    
    print("\nPOURQUOI LE TGN NE FAIT PAS AUC=1.0:")
    print("   - Le TGN apprend des embeddings complexes")
    print("   - Il n'a peut-être pas encore appris cette règle simple")
    print("   - Ou il overfitte sur d'autres patterns moins utiles")
    
    print("\n" + "="*70)
    print("SOLUTIONS POSSIBLES")
    print("="*70)
    print("\n[OK] SOLUTION 1: Négatives plus réalistes")
    print("   Générer des négatives qui ont aussi raised>0 et rounds>0")
    print("   → Sampler parmi les edges EXISTANTS mais à un autre timestamp")
    
    print("\n[OK] SOLUTION 2: Ne pas utiliser raised et rounds comme features")
    print("   Utiliser UNIQUEMENT les degrés: [u_deg, v_deg]")
    print("   → Test si le problème persiste")
    
    print("\n[OK] SOLUTION 3: Negative sampling temporel")
    print("   Pour chaque positive au temps t, créer une négative")
    print("   en prenant un edge qui existe à t-1 mais pas à t")
    
    print("\n[OK] SOLUTION 4: Vérifier ce que fait réellement TGN")
    print("   Inspecter les edge_features et node_features du TGN")
    print("   → Voir s'il a accès aux mêmes informations")


# ==============================
# SOLUTION : RF avec features réduites
# ==============================
def test_rf_degree_only():
    """
    Tester RF avec SEULEMENT les degrés (pas raised/rounds).
    """
    print("\n" + "="*70)
    print("TEST : RF avec SEULEMENT les degrés")
    print("="*70)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    # Charger le graphe
    with open(GRAPH_PATH, "rb") as f:
        B = pickle.load(f)
    
    df = pd.read_csv(DATA_CSV)
    val_time, test_time = list(np.quantile(df.ts, [0.70, 0.85]))
    
    # Extraire edges
    train_edges, test_edges = [], []
    for u, v, data in B.edges(data=True):
        ts_list = []
        for fr in data.get("funding_rounds", []):
            ann = fr.get("announced_on", None)
            if ann:
                try:
                    ts_list.append(pd.to_datetime(ann).timestamp())
                except:
                    continue
        if not ts_list:
            continue
        first_ts = min(ts_list)
        if first_ts <= val_time:
            train_edges.append((u, v, 1, first_ts))
        elif first_ts > test_time:
            test_edges.append((u, v, 1, first_ts))
    
    nodes_comp = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    nodes_inv = [n for n, d in B.nodes(data=True) if d["bipartite"] == 1]
    
    # Générer négatives
    from utils.utils import RandEdgeSampler
    train_rand_sampler = RandEdgeSampler(
        [u for u, v, _, _ in train_edges],
        [v for u, v, _, _ in train_edges]
    )
    test_rand_sampler = RandEdgeSampler(
        [u for u, v, _, _ in train_edges + test_edges],
        [v for u, v, _, _ in train_edges + test_edges]
    )
    
    _, train_neg_invs = train_rand_sampler.sample(len(train_edges))
    _, test_neg_invs = test_rand_sampler.sample(len(test_edges))
    
    train_neg = [(nodes_comp[i % len(nodes_comp)], inv, 0, 0) 
                 for i, inv in enumerate(train_neg_invs)]
    test_neg = [(nodes_comp[i % len(nodes_comp)], inv, 0, 0) 
                for i, inv in enumerate(test_neg_invs)]
    
    train_all = train_edges + train_neg
    test_all = test_edges + test_neg
    
    # Features: SEULEMENT les degrés
    def degree_at_time(B, node, cutoff_ts):
        deg = 0
        for nbr in B.neighbors(node):
            edge_data = B.get_edge_data(node, nbr)
            if not edge_data:
                continue
            ts_list = []
            for fr in edge_data.get("funding_rounds", []):
                ann = fr.get("announced_on", None)
                if ann:
                    try:
                        ts_list.append(pd.to_datetime(ann).timestamp())
                    except:
                        continue
            if ts_list and min(ts_list) <= cutoff_ts:
                deg += 1
        return deg
    
    X_train, y_train = [], []
    for u, v, label, ts in train_all:
        u_deg = degree_at_time(B, u, ts)
        v_deg = degree_at_time(B, v, ts)
        X_train.append([u_deg, v_deg])  # SEULEMENT degrés
        y_train.append(label)
    
    X_test, y_test = [], []
    for u, v, label, ts in test_all:
        u_deg = degree_at_time(B, u, ts)
        v_deg = degree_at_time(B, v, ts)
        X_test.append([u_deg, v_deg])
        y_test.append(label)
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # Train RF
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_test)[:, 1]
    
    print(f"\n🔵 RF avec SEULEMENT [u_deg, v_deg]:")
    print(f"  AUC = {roc_auc_score(y_test, y_pred):.4f}")
    print(f"  AP  = {average_precision_score(y_test, y_pred):.4f}")
    
    print("\nSi AUC < 1.0, alors le problème était bien raised/rounds!")
    print("   Si AUC ≈ 1.0, alors même les degrés sont trop discriminants")


if __name__ == "__main__":
    diagnostic_data_leakage()
    print("\n\n")
    test_rf_degree_only()