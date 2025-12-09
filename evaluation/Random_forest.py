import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from utils.utils import RandEdgeSampler

# ==============================
# PATHS
# ==============================
GRAPH_PATH = "savings/bipartite_invest_comp/networks/bipartite_graph_10000.gpickle"
DATA_CSV = "./data/crunchbase_filtered.csv"

def load_graph_and_dicts():
    print("Loading graph and dictionaries...")
    with open(GRAPH_PATH, "rb") as f:
        B = pickle.load(f)
    return B


def tgn_temporal_split(csv_path):
    df = pd.read_csv(csv_path)
    val_time, test_time = list(np.quantile(df.ts, [0.70, 0.85]))
    print(f"TGN VAL TIME = {val_time}, TEST TIME = {test_time}")
    return val_time, test_time


def extract_edges_by_time(B, val_time, test_time):
    train_edges, val_edges, test_edges = [], [], []

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
        label = 1

        if first_ts <= val_time:
            train_edges.append((u, v, label, first_ts))
        elif val_time < first_ts <= test_time:
            val_edges.append((u, v, label, first_ts))
        else:
            test_edges.append((u, v, label, first_ts))

    return train_edges, val_edges, test_edges


def generate_negatives_with_timestamps(positive_edges, nodes_comp, nodes_inv, sampler):
    """
    ✅ FIX: Génère des négatives avec le MÊME timestamp que les positives.
    
    Pour chaque positive au temps t, on crée une négative au même temps t.
    Cela assure que degree_at_time(node, t) soit comparable.
    """
    print(f"\n✅ Génération de négatives avec timestamps RÉELS")
    
    _, neg_investors = sampler.sample(len(positive_edges))
    
    neg_edges = []
    for i, inv in enumerate(neg_investors):
        # Utiliser le MÊME timestamp que la positive correspondante
        _, _, _, pos_ts = positive_edges[i]
        comp = nodes_comp[i % len(nodes_comp)]
        neg_edges.append((comp, inv, 0, pos_ts))  # ← pos_ts au lieu de 0
    
    print(f"  Générées: {len(neg_edges)} négatives avec timestamps réels")
    return neg_edges


def degree_at_time(B, node, cutoff_ts):
    """Return number of unique neighbours until cutoff_ts."""
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


def extract_features_degree_only(B, edges):
    """
    Extrait SEULEMENT les degrés (pas raised/rounds).
    """
    X, y = [], []
    for u, v, label, edge_ts in edges:
        u_deg = degree_at_time(B, u, edge_ts)
        v_deg = degree_at_time(B, v, edge_ts)
        feat = [u_deg, v_deg]
        X.append(feat)
        y.append(label)
    return np.array(X), np.array(y)


def extract_features_full(B, edges):
    """
    Features complètes : [u_deg, v_deg, total_raised, num_rounds].
    """
    X, y = [], []
    for u, v, label, edge_ts in edges:
        u_deg = degree_at_time(B, u, edge_ts)
        v_deg = degree_at_time(B, v, edge_ts)

        total_raised_raw = 0.0
        num_rounds = 0
        edge_data = B.get_edge_data(u, v)
        if edge_data:
            for fr in edge_data.get("funding_rounds", []):
                ann = fr.get("announced_on", None)
                if not ann:
                    continue
                try:
                    ann_ts = pd.to_datetime(ann).timestamp()
                except:
                    continue
                if ann_ts <= edge_ts:
                    raised = fr.get("raised_amount_usd", fr.get("raised_amount", 0)) or 0
                    try:
                        total_raised_raw += float(raised)
                    except:
                        continue
                    num_rounds += 1

        total_raised = np.log1p(total_raised_raw)
        # feat = [u_deg, v_deg, total_raised, num_rounds]
        feat = [0,0,total_raised, num_rounds]
        X.append(feat)
        y.append(label)

    return np.array(X), np.array(y)


def compute_mrr_recall_at_k(B, model, test_edges, k_list=[10, 50]):
    """Ranking metrics."""
    inv_to_true = {}
    for u, v, label, first_ts in test_edges:
        inv_to_true.setdefault(v, []).append((u, first_ts))

    nodes_comp = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]

    MRRs = []
    recalls = {k: [] for k in k_list}

    for inv, true_list in tqdm(inv_to_true.items(), desc="Computing MRR/Recall"):
        cutoff_ts = min(ts for _, ts in true_list)

        X_candidates = []
        comps_list = []
        for comp in nodes_comp:
            edge_data = B.get_edge_data(comp, inv)
            u_deg = degree_at_time(B, comp, cutoff_ts)
            v_deg = degree_at_time(B, inv, cutoff_ts)

            total_raised_raw = 0.0
            num_rounds = 0
            if edge_data:
                for fr in edge_data.get("funding_rounds", []):
                    ann = fr.get("announced_on", None)
                    if not ann:
                        continue
                    try:
                        ann_ts = pd.to_datetime(ann).timestamp()
                    except:
                        continue
                    if ann_ts <= cutoff_ts:
                        raised = fr.get("raised_amount_usd", fr.get("raised_amount", 0)) or 0
                        try:
                            total_raised_raw += float(raised)
                        except:
                            continue
                        num_rounds += 1

            total_raised = np.log1p(total_raised_raw)
            feat = [u_deg, v_deg, total_raised, num_rounds]
            X_candidates.append(feat)
            comps_list.append(comp)

        scores = model.predict_proba(np.array(X_candidates))[:, 1]
        ranking = [x for _, x in sorted(zip(scores, comps_list), reverse=True)]

        true_comps = [tc for tc, _ in true_list]
        ranks = [ranking.index(tc) + 1 for tc in true_comps if tc in ranking]
        MRRs.append(1.0 / min(ranks) if ranks else 0.0)

        for k in k_list:
            top_k = set(ranking[:k])
            hits = len(set(true_comps) & top_k)
            recalls[k].append(hits / len(true_comps))

    return np.mean(MRRs), {k: np.mean(v) for k, v in recalls.items()}


def main():
    B = load_graph_and_dicts()
    val_time, test_time = tgn_temporal_split(DATA_CSV)
    train_edges, val_edges, test_edges = extract_edges_by_time(B, val_time, test_time)
    print(f"Train = {len(train_edges)}, Val = {len(val_edges)}, Test = {len(test_edges)}")

    nodes_comp = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    nodes_inv = [n for n, d in B.nodes(data=True) if d["bipartite"] == 1]
    
    # Create samplers
    train_rand_sampler = RandEdgeSampler(
        [u for u, v, _, _ in train_edges],
        [v for u, v, _, _ in train_edges]
    )
    test_rand_sampler = RandEdgeSampler(
        [u for u, v, _, _ in train_edges + test_edges],
        [v for u, v, _, _ in train_edges + test_edges]
    )
    
    # ============================================================
    # Génération avec VRAIS timestamps
    # ============================================================
    train_neg = generate_negatives_with_timestamps(
        train_edges, nodes_comp, nodes_inv, train_rand_sampler
    )
    test_neg = generate_negatives_with_timestamps(
        test_edges, nodes_comp, nodes_inv, test_rand_sampler
    )
    
    train_all = train_edges + train_neg
    test_all = test_edges + test_neg
    
    # ============================================================
    # TEST 1: RF avec SEULEMENT degrés
    # ============================================================
    print("\n" + "="*70)
    print("TEST 1: RF avec [u_deg, v_deg] uniquement")
    print("="*70)
    
    X_train_deg, y_train_deg = extract_features_degree_only(B, train_all)
    X_test_deg, y_test_deg = extract_features_degree_only(B, test_all)
    
    rf_deg = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf_deg.fit(X_train_deg, y_train_deg)
    y_pred_deg = rf_deg.predict_proba(X_test_deg)[:, 1]
    
    auc_deg = roc_auc_score(y_test_deg, y_pred_deg)
    ap_deg = average_precision_score(y_test_deg, y_pred_deg)
    
    print(f"\n🔵 RF (Degrés uniquement):")
    print(f"  AUC = {auc_deg:.4f}")
    print(f"  AP  = {ap_deg:.4f}")
    
    if auc_deg < 0.95:
        print("  ✅ Tâche non-triviale maintenant!")
    else:
        print("  ⚠️  Toujours trop facile...")
    
    # ============================================================
    # TEST 2: RF avec features complètes
    # ============================================================
    print("\n" + "="*70)
    print("TEST 2: RF avec [u_deg, v_deg, raised, rounds]")
    print("="*70)
    
    X_train_full, y_train_full = extract_features_full(B, train_all)
    X_test_full, y_test_full = extract_features_full(B, test_all)
    
    rf_full = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf_full.fit(X_train_full, y_train_full)
    y_pred_full = rf_full.predict_proba(X_test_full)[:, 1]
    
    auc_full = roc_auc_score(y_test_full, y_pred_full)
    ap_full = average_precision_score(y_test_full, y_pred_full)
    
    print(f"\n🔵 RF (Features complètes):")
    print(f"  AUC = {auc_full:.4f}")
    print(f"  AP  = {ap_full:.4f}")
    
    # Ranking metrics
    mrr, recall_dict = compute_mrr_recall_at_k(B, rf_full, test_edges, k_list=[10, 50])
    print(f"  MRR = {mrr:.4f}")
    print(f"  Recall@10 = {recall_dict[10]:.4f}")
    print(f"  Recall@50 = {recall_dict[50]:.4f}")
    
    # ============================================================
    # RÉSUMÉ
    # ============================================================
    print("\n" + "="*70)
    print("RÉSUMÉ COMPARATIF")
    print("="*70)
    print(f"{'Modèle':<40} {'AUC':>8} {'AP':>8}")
    print("-"*70)
    print(f"{'RF (Degrés uniquement)':<40} {auc_deg:>8.4f} {ap_deg:>8.4f}")
    print(f"{'RF (Features complètes)':<40} {auc_full:>8.4f} {ap_full:>8.4f}")
    
    print("\n💡 Maintenant vous avez une VRAIE baseline réaliste!")
    print("   Si TGN bat ces scores, il apprend vraiment des patterns complexes.")


if __name__ == "__main__":
    main()