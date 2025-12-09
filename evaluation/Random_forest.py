import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# ==============================
# PATHS
# ==============================
GRAPH_PATH = "savings/bipartite_invest_comp/networks/bipartite_graph_10000.gpickle"
DICT_COMP_PATH = "savings/bipartite_invest_comp/classes/dict_companies_10000.pickle"
DICT_INV_PATH = "savings/bipartite_invest_comp/classes/dict_investors_10000.pickle"
DATA_CSV = "./data/crunchbase_filtered.csv"     # <-- même CSV que pour TGN
NEGATIVE_SAMPLE_RATIO = 1

# ==============================
# LOAD GRAPH + DICTS
# ==============================
def load_graph_and_dicts():
    print("Loading graph and dictionaries...")
    with open(GRAPH_PATH, "rb") as f:
        B = pickle.load(f)
    with open(DICT_COMP_PATH, "rb") as f:
        dict_comp = pickle.load(f)
    with open(DICT_INV_PATH, "rb") as f:
        dict_inv = pickle.load(f)
    return B, dict_comp, dict_inv


# ==============================
# SPLIT TEMPOREL VERSION TGN
# ==============================
def tgn_temporal_split(csv_path):
    df = pd.read_csv(csv_path)

    # timestamps
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


# ==============================
# NEGATIVE SAMPLING
# ==============================
def generate_negative_samples(B, existing_edges, ratio=1):
    nodes_comp = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]
    nodes_inv = [n for n, d in B.nodes(data=True) if d["bipartite"] == 1]

    existing_pairs = set((u, v) for u, v, _, _ in existing_edges)

    neg_edges = []
    pbar = tqdm(total=len(existing_edges) * ratio, desc="Generating negatives")

    while len(neg_edges) < len(existing_edges) * ratio:
        u = np.random.choice(nodes_comp)
        v = np.random.choice(nodes_inv)
        if (u, v) not in existing_pairs:
            neg_edges.append((u, v, 0, 0))
            pbar.update(1)

    pbar.close()
    return neg_edges


# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_features(B, edges):
    X, y = [], []
    for u, v, label, _ in edges:
        u_deg = B.degree(u)
        v_deg = B.degree(v)

        edge_data = B.get_edge_data(u, v)

        # LOG TRANSFORM ✔
        total_raised_raw = edge_data.get("total_raised_amount_usd", 0) if edge_data else 0
        total_raised = np.log1p(total_raised_raw)

        num_rounds = edge_data.get("num_funding_rounds", 1) if edge_data else 1

        feat = [u_deg, v_deg, total_raised, num_rounds]
        X.append(feat)
        y.append(label)

    return np.array(X), np.array(y)


# ==============================
# MRR + RECALL@K
# ==============================
def compute_mrr_recall_at_k(B, model, test_edges, k_list=[10, 50]):
    inv_to_true = {}
    for u, v, label, _ in test_edges:
        inv_to_true.setdefault(v, []).append(u)

    nodes_comp = [n for n, d in B.nodes(data=True) if d["bipartite"] == 0]

    MRRs = []
    recalls = {k: [] for k in k_list}

    for inv, true_comps in tqdm(inv_to_true.items(), desc="Computing MRR/Recall"):
        X_candidates = []
        comps_list = []

        for comp in nodes_comp:
            edge_data = B.get_edge_data(comp, inv)

            u_deg = B.degree(comp)
            v_deg = B.degree(inv)

            # LOG TRANSFORM ✔
            total_raised_raw = edge_data.get("total_raised_amount_usd", 0) if edge_data else 0
            total_raised = np.log1p(total_raised_raw)

            num_rounds = edge_data.get("num_funding_rounds", 1) if edge_data else 1

            feat = [u_deg, v_deg, total_raised, num_rounds]
            X_candidates.append(feat)
            comps_list.append(comp)

        scores = model.predict_proba(np.array(X_candidates))[:, 1]
        ranking = [x for _, x in sorted(zip(scores, comps_list), reverse=True)]

        # MRR
        ranks = [ranking.index(tc) + 1 for tc in true_comps if tc in ranking]
        if ranks:
            MRRs.append(1.0 / min(ranks))
        else:
            MRRs.append(0.0)

        # Recall@k
        for k in k_list:
            top_k = set(ranking[:k])
            hit = len(set(true_comps) & top_k)
            recalls[k].append(hit / len(true_comps))

    return np.mean(MRRs), {k: np.mean(v) for k, v in recalls.items()}


# ==============================
# MAIN
# ==============================
def main():
    B, dict_comp, dict_inv = load_graph_and_dicts()

    # SAME SPLIT AS TGN ✔
    val_time, test_time = tgn_temporal_split(DATA_CSV)

    train_edges, val_edges, test_edges = extract_edges_by_time(B, val_time, test_time)
    print(f"Train = {len(train_edges)}, Val = {len(val_edges)}, Test = {len(test_edges)}")

    # Negative sampling
    train_neg = generate_negative_samples(B, train_edges, NEGATIVE_SAMPLE_RATIO)
    test_neg = generate_negative_samples(B, test_edges, NEGATIVE_SAMPLE_RATIO)

    train_all = train_edges + train_neg
    test_all = test_edges + test_neg

    # Feature extraction
    X_train, y_train = extract_features(B, train_all)
    X_test, y_test = extract_features(B, test_all)

    # Train RF
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Predictions
    y_pred_prob = rf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_prob)
    ap = average_precision_score(y_test, y_pred_prob)

    print("\n=====================================")
    print("   RANDOM FOREST — TGN SPLIT")
    print("=====================================")
    print(f"AUC = {auc:.4f}")
    print(f"AP  = {ap:.4f}")

    # Ranking metrics identical to TGN ✔
    mrr, recall_dict = compute_mrr_recall_at_k(B, rf, test_edges, k_list=[10, 50])
    print(f"MRR = {mrr:.4f}")
    print(f"Recall@10 = {recall_dict[10]:.4f}")
    print(f"Recall@50 = {recall_dict[50]:.4f}")


if __name__ == "__main__":
    main()
