import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import warnings
import pickle
warnings.filterwarnings('ignore')

# ==============================
# PATHS
# ==============================
TGN_EDGES_PATH = "data/crunchbase_filtered.csv"
COMPANY_MAP_PATH = "data/mappings/crunchbase_filtered_company_id_map.pickle"
INVESTOR_MAP_PATH = "data/mappings/crunchbase_filtered_investor_id_map.pickle"
RAW_DATA_PATH = "debug_df_graphcaca21.csv"

def load_data():
    """Charge toutes les données nécessaires"""
    print("Chargement des données...")
    
    tgn_edges = pd.read_csv(TGN_EDGES_PATH)
    print(f"  Edges TGN chargées: {len(tgn_edges):,}")
    
    raw_data = pd.read_csv(RAW_DATA_PATH)
    print(f"  Données brutes chargées: {len(raw_data):,}")
    
    with open(COMPANY_MAP_PATH, 'rb') as f:
        company_map = pickle.load(f)
    
    with open(INVESTOR_MAP_PATH, 'rb') as f:
        investor_map = pickle.load(f)
    
    id_to_company = {v: k for k, v in company_map.items()}
    id_to_investor = {v: k for k, v in investor_map.items()}
    
    return tgn_edges, raw_data, id_to_company, id_to_investor

def create_temporal_splits(tgn_edges):
    """Crée des splits temporels pour prédiction FUTURE"""
    tgn_edges = tgn_edges.sort_values('ts')
    
    train_cutoff = tgn_edges['ts'].quantile(0.70)
    val_cutoff = tgn_edges['ts'].quantile(0.85)
    
    print(f"\n📅 Split temporel:")
    print(f"  Train: ts ≤ {train_cutoff}")
    print(f"  Val:   {train_cutoff} < ts ≤ {val_cutoff}")
    print(f"  Test:  ts > {val_cutoff}")
    
    tgn_edges['datetime'] = pd.to_datetime(tgn_edges['ts'], unit='s')
    
    return train_cutoff, val_cutoff, tgn_edges

def compute_company_features(raw_data, company, timestamp):
    """Calcule les features d'une compagnie JUSQU'À timestamp"""
    company_data = raw_data[
        (raw_data['org_name'] == company) & 
        (pd.to_datetime(raw_data['announced_on']) <= pd.to_datetime(timestamp, unit='s'))
    ]
    
    if company_data.empty:
        return {
            'total_raised': 0,
            'num_rounds': 0,
            'num_investors': 0,
            'avg_round_size': 0,
            'days_since_last_round': 365*5
        }
    
    total_raised = company_data['raised_amount_usd'].sum()
    num_rounds = len(company_data)
    num_investors = company_data['investor_name'].nunique() # comptage du nombre d'investisseurs uniques
    avg_round_size = total_raised / num_rounds if num_rounds > 0 else 0
    
    last_round_date = pd.to_datetime(company_data['announced_on']).max()
    current_date = pd.to_datetime(timestamp, unit='s')
    days_since_last = (current_date - last_round_date).days
    
    return {
        'total_raised': total_raised,
        'num_rounds': num_rounds,
        'num_investors': num_investors,
        'avg_round_size': avg_round_size,
        'days_since_last_round': days_since_last
    }

def compute_investor_features(raw_data, investor, timestamp):
    """Calcule les features d'un investisseur JUSQU'À timestamp"""
    investor_data = raw_data[
        (raw_data['investor_name'] == investor) & 
        (pd.to_datetime(raw_data['announced_on']) <= pd.to_datetime(timestamp, unit='s'))
    ]
    
    if investor_data.empty:
        return {
            'total_invested': 0,
            'num_investments': 0,
            'num_companies': 0,
            'avg_investment_size': 0,
            'days_since_last_investment': 365*2
        }
    
    total_invested = investor_data['raised_amount_usd'].sum()
    num_investments = len(investor_data)
    num_companies = investor_data['org_name'].nunique()
    avg_investment_size = total_invested / num_investments if num_investments > 0 else 0
    
    last_invest_date = pd.to_datetime(investor_data['announced_on']).max()
    current_date = pd.to_datetime(timestamp, unit='s')
    days_since_last = (current_date - last_invest_date).days
    
    return {
        'total_invested': total_invested,
        'num_investments': num_investments,
        'num_companies': num_companies,
        'avg_investment_size': avg_investment_size,
        'days_since_last_investment': days_since_last
    }

def compute_pair_features(raw_data, company, investor, timestamp):
    """Calcule les features de la paire basées sur l'historique COMMUN"""
    investor_history = raw_data[
        (raw_data['investor_name'] == investor) & 
        (pd.to_datetime(raw_data['announced_on']) <= pd.to_datetime(timestamp, unit='s'))
    ]
    
    investor_companies = investor_history['org_name'].unique()
    investor_categories = set()
    
    for comp in investor_companies:
        comp_data = raw_data[raw_data['org_name'] == comp]
        if not comp_data.empty and 'category_list' in comp_data.columns:
            cats = comp_data['category_list'].iloc[0]
            if isinstance(cats, str):
                investor_categories.update([c.strip() for c in cats.split(',')])
    
    company_data = raw_data[raw_data['org_name'] == company]
    company_categories = set()
    if not company_data.empty and 'category_list' in company_data.columns:
        cats = company_data['category_list'].iloc[0]
        if isinstance(cats, str):
            company_categories.update([c.strip() for c in cats.split(',')])
    
    if investor_categories and company_categories:
        category_overlap = len(investor_categories & company_categories) / len(investor_categories | company_categories)
    else:
        category_overlap = 0
    
    company_investors = set(raw_data[
        (raw_data['org_name'] == company) & 
        (pd.to_datetime(raw_data['announced_on']) <= pd.to_datetime(timestamp, unit='s'))
    ]['investor_name'].unique())
    
    common_co_investors = 0
    for other_inv in company_investors:
        if other_inv == investor:
            continue
        other_inv_companies = set(raw_data[
            (raw_data['investor_name'] == other_inv) & 
            (pd.to_datetime(raw_data['announced_on']) <= pd.to_datetime(timestamp, unit='s'))
        ]['org_name'].unique())
        
        investor_companies_set = set(investor_companies)
        if investor_companies_set & other_inv_companies:
            common_co_investors += 1
    
    return {
        'category_overlap': category_overlap,
        'common_co_investors': common_co_investors,
        'investor_experience_with_similar': len(investor_companies)
    }

def extract_features(raw_data, company, investor, timestamp):
    """Extract features for a company-investor pair"""
    company_feats = compute_company_features(raw_data, company, timestamp)
    investor_feats = compute_investor_features(raw_data, investor, timestamp)
    pair_feats = compute_pair_features(raw_data, company, investor, timestamp)
    
    features = [
        np.log1p(company_feats['total_raised']),
        np.log1p(company_feats['num_rounds'] + 1),
        company_feats['num_investors'],
        np.log1p(company_feats['avg_round_size']),
        np.log1p(company_feats['days_since_last_round'] + 1),
        
        np.log1p(investor_feats['total_invested']),
        np.log1p(investor_feats['num_investments'] + 1),
        investor_feats['num_companies'],
        np.log1p(investor_feats['avg_investment_size']),
        np.log1p(investor_feats['days_since_last_investment'] + 1),
        
        pair_feats['category_overlap'],
        pair_feats['common_co_investors'],
        np.log1p(pair_feats['investor_experience_with_similar'] + 1)
    ]
    
    return features

def create_positive_samples(tgn_edges, raw_data, id_to_company, id_to_investor, split_type='test'):
    """Crée des échantillons POSITIFS (label=1) pour la prédiction FUTURE"""
    positives = []
    
    if split_type == 'train':
        edges = tgn_edges[tgn_edges['ts'] <= train_cutoff]
    elif split_type == 'val':
        edges = tgn_edges[(tgn_edges['ts'] > train_cutoff) & (tgn_edges['ts'] <= val_cutoff)]
    else:
        edges = tgn_edges[tgn_edges['ts'] > val_cutoff]
    
    print(f"\nCréation des positifs ({split_type}): {len(edges)} edges")
    
    for idx, row in tqdm(edges.iterrows(), total=len(edges), desc=f"Processing {split_type} positives"):
        company = id_to_company.get(int(row['u']), f"company_{row['u']}")
        investor = id_to_investor.get(int(row['i']), f"investor_{row['i']}")
        timestamp = row['ts']
        
        features = extract_features(raw_data, company, investor, timestamp)
        
        positives.append({
            'company': company,
            'investor': investor,
            'timestamp': timestamp,
            'features': features,
            'label': 1
        })
    
    return positives

def create_negative_samples(positives, raw_data, num_negatives_per_positive=1):
    """Crée des échantillons NÉGATIFS (label = 0) plausibles"""
    negatives = []
    
    all_companies = raw_data['org_name'].unique()
    all_investors = raw_data['investor_name'].unique()
    
    print(f"\nCréation des négatifs: {len(positives)} × {num_negatives_per_positive}")
    
    for pos in tqdm(positives, desc="Generating negatives"):
        company = pos['company']
        investor = pos['investor']
        timestamp = pos['timestamp']
        
        funded_companies = set(raw_data[
            (raw_data['investor_name'] == investor) & 
            (pd.to_datetime(raw_data['announced_on']) <= pd.to_datetime(timestamp, unit='s'))
        ]['org_name'].unique())
        
        candidate_companies = [c for c in all_companies 
                              if c not in funded_companies and c != company]
        
        num_to_sample = min(num_negatives_per_positive, len(candidate_companies))
        if num_to_sample > 0:
            negative_companies = np.random.choice(candidate_companies, size=num_to_sample, replace=False)
            
            for neg_company in negative_companies:
                features = extract_features(raw_data, neg_company, investor, timestamp)
                
                negatives.append({
                    'company': neg_company,
                    'investor': investor,
                    'timestamp': timestamp,
                    'features': features,
                    'label': 0
                })
    
    return negatives

def create_ranking_samples(tgn_edges, raw_data, id_to_company, id_to_investor, split_type='test', num_negatives=100):
    """
    Crée des échantillons pour l'évaluation RANKING (MRR, Recall@K)
    Pour chaque positive edge, on crée K candidats négatifs
    """
    ranking_samples = []
    
    if split_type == 'train':
        edges = tgn_edges[tgn_edges['ts'] <= train_cutoff]
    elif split_type == 'val':
        edges = tgn_edges[(tgn_edges['ts'] > train_cutoff) & (tgn_edges['ts'] <= val_cutoff)]
    else:
        edges = tgn_edges[tgn_edges['ts'] > val_cutoff]
    
    all_companies = raw_data['org_name'].unique()
    
    print(f"\nCréation des samples de RANKING ({split_type}): {len(edges)} queries")
    
    for idx, row in tqdm(edges.iterrows(), total=len(edges), desc=f"Processing {split_type} ranking"):
        true_company = id_to_company.get(int(row['u']), f"company_{row['u']}")
        investor = id_to_investor.get(int(row['i']), f"investor_{row['i']}")
        timestamp = row['ts']
        
        # Historique de l'investisseur
        funded_companies = set(raw_data[
            (raw_data['investor_name'] == investor) & 
            (pd.to_datetime(raw_data['announced_on']) <= pd.to_datetime(timestamp, unit='s'))
        ]['org_name'].unique())
        
        # Candidats négatifs: compagnies non financées
        candidate_companies = [c for c in all_companies 
                              if c not in funded_companies and c != true_company]
        
        if len(candidate_companies) < num_negatives:
            num_negatives_sample = len(candidate_companies)
        else:
            num_negatives_sample = num_negatives
        
        if num_negatives_sample > 0:
            negative_companies = np.random.choice(candidate_companies, 
                                                  size=num_negatives_sample, 
                                                  replace=False)
            
            # Créer un groupe de ranking: 1 positive + K negatives
            candidates = []
            
            # Positive (label=1)
            pos_features = extract_features(raw_data, true_company, investor, timestamp)
            candidates.append({
                'company': true_company,
                'investor': investor,
                'features': pos_features,
                'label': 1
            })
            
            # Negatives (label=0)
            for neg_company in negative_companies:
                neg_features = extract_features(raw_data, neg_company, investor, timestamp)
                candidates.append({
                    'company': neg_company,
                    'investor': investor,
                    'features': neg_features,
                    'label': 0
                })
            
            ranking_samples.append({
                'investor': investor,
                'timestamp': timestamp,
                'candidates': candidates
            })
    
    return ranking_samples

def compute_ranking_metrics(model, ranking_samples, k_values=[10, 50]):
    """
    Calcule MRR, Recall@K pour les échantillons de ranking
    """
    mrr_scores = []
    recall_at_k = {k: [] for k in k_values}
    
    print(f"\nCalcul des métriques de ranking sur {len(ranking_samples)} queries...")
    
    for sample in tqdm(ranking_samples, desc="Computing ranking metrics"):
        candidates = sample['candidates']
        
        # Extraire features et labels
        X = np.array([c['features'] for c in candidates])
        y_true = np.array([c['label'] for c in candidates])
        
        # Prédire les scores
        y_scores = model.predict_proba(X)[:, 1]
        
        # Trier par score décroissant
        sorted_indices = np.argsort(-y_scores)
        sorted_labels = y_true[sorted_indices]
        
        # MRR: position du premier vrai positif
        positive_positions = np.where(sorted_labels == 1)[0]
        if len(positive_positions) > 0:
            rank = positive_positions[0] + 1  # +1 car on compte à partir de 1
            mrr_scores.append(1.0 / rank)
        else:
            mrr_scores.append(0.0)
        
        # Recall@K
        for k in k_values:
            top_k_labels = sorted_labels[:k]
            if np.sum(sorted_labels) > 0:  # Si au moins un positif existe
                recall = np.sum(top_k_labels) / np.sum(sorted_labels)
                recall_at_k[k].append(recall)
    
    # Moyennes
    mrr = np.mean(mrr_scores)
    recall_k = {k: np.mean(scores) for k, scores in recall_at_k.items()}
    
    return mrr, recall_k

def prepare_dataset(positives, negatives):
    """Prépare les matrices X et y pour l'entraînement"""
    all_samples = positives + negatives
    np.random.shuffle(all_samples)
    
    X = np.array([s['features'] for s in all_samples])
    y = np.array([s['label'] for s in all_samples])
    
    print(f"\nDataset final:")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Positives: {len(positives)} ({100*len(positives)/len(all_samples):.1f}%)")
    print(f"  Negatives: {len(negatives)} ({100*len(negatives)/len(all_samples):.1f}%)")
    print(f"  Features: {X.shape[1]}")
    
    return X, y

def main():
    print("="*80)
    print("RANDOM FOREST BASELINE - Toutes les métriques (AUROC, AP, MRR, Recall@K)")
    print("="*80)
    
    # 1. Charger les données
    tgn_edges, raw_data, id_to_company, id_to_investor = load_data()
    
    # 2. Créer les splits temporels
    global train_cutoff, val_cutoff
    train_cutoff, val_cutoff, tgn_edges = create_temporal_splits(tgn_edges)
    
    # 3. Préparer les données brutes
    raw_data['announced_on'] = pd.to_datetime(raw_data['announced_on'], errors='coerce')
    raw_data = raw_data.dropna(subset=['announced_on', 'org_name', 'investor_name'])
    
    # 4. Créer les datasets pour CLASSIFICATION (AUROC, AP)
    print("\n" + "="*80)
    print("CRÉATION DES DATASETS - CLASSIFICATION")
    print("="*80)

    # TRAIN
    train_positives = create_positive_samples(tgn_edges, raw_data, id_to_company, id_to_investor, 'train')
    train_negatives = create_negative_samples(train_positives, raw_data, num_negatives_per_positive=1)
    X_train, y_train = prepare_dataset(train_positives, train_negatives)

    # VALIDATION
    val_positives = create_positive_samples(tgn_edges, raw_data, id_to_company, id_to_investor, 'val')
    val_negatives = create_negative_samples(val_positives, raw_data, num_negatives_per_positive=1)
    X_val, y_val = prepare_dataset(val_positives, val_negatives)

    # TEST
    test_positives = create_positive_samples(tgn_edges, raw_data, id_to_company, id_to_investor, 'test')
    test_negatives = create_negative_samples(test_positives, raw_data, num_negatives_per_positive=1)
    X_test, y_test = prepare_dataset(test_positives, test_negatives)

    # 5. Créer les datasets pour RANKING (MRR, Recall@K)
    print("\n" + "="*80)
    print("CRÉATION DES DATASETS - RANKING")
    print("="*80)

    val_ranking_samples = create_ranking_samples(
        tgn_edges, raw_data, id_to_company, id_to_investor,
        split_type='val', num_negatives=100
    )

    test_ranking_samples = create_ranking_samples(
        tgn_edges, raw_data, id_to_company, id_to_investor,
        split_type='test', num_negatives=100
    )
    
    # 6. Entraînement
    print("\n" + "="*80)
    print("ENTRAÎNEMENT DU RANDOM FOREST")
    print("="*80)
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    print("\nEntraînement en cours...")
    rf.fit(X_train, y_train)

    # 7. Évaluation - VALIDATION SET
    print("\n" + "="*80)
    print("ÉVALUATION - VALIDATION SET")
    print("="*80)

    # Classification metrics sur validation
    y_val_pred_proba = rf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    val_ap = average_precision_score(y_val, y_val_pred_proba)

    print(f"\nMétriques de Classification (Validation):")
    print(f"  AUROC:               {val_auc:.4f}")
    print(f"  Average Precision:   {val_ap:.4f}")

    # Ranking metrics sur validation
    val_mrr, val_recall_k = compute_ranking_metrics(rf, val_ranking_samples, k_values=[10, 50])

    print(f"\nMétriques de Ranking (Validation):")
    print(f"  MRR:                 {val_mrr:.4f}")
    print(f"  Recall@10:           {val_recall_k[10]:.4f}")
    print(f"  Recall@50:           {val_recall_k[50]:.4f}")

    # 8. Évaluation - TEST SET
    print("\n" + "="*80)
    print("ÉVALUATION - TEST SET")
    print("="*80)

    # Classification metrics sur test
    y_test_pred_proba = rf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    test_ap = average_precision_score(y_test, y_test_pred_proba)

    print(f"\nMétriques de Classification (Test):")
    print(f"  AUROC:               {test_auc:.4f}")
    print(f"  Average Precision:   {test_ap:.4f}")

    # Ranking metrics sur test
    test_mrr, test_recall_k = compute_ranking_metrics(rf, test_ranking_samples, k_values=[10, 50])

    print(f"\nMétriques de Ranking (Test):")
    print(f"  MRR:                 {test_mrr:.4f}")
    print(f"  Recall@10:           {test_recall_k[10]:.4f}")
    print(f"  Recall@50:           {test_recall_k[50]:.4f}")
    
    # 9. Analyse des features
    print("\n" + "="*80)
    print("ANALYSE DES FEATURES")
    print("="*80)
    
    feature_names = [
        'log_company_total_raised',
        'log_company_num_rounds',
        'company_num_investors',
        'log_company_avg_round_size',
        'log_company_days_since_last',
        
        'log_investor_total_invested',
        'log_investor_num_investments',
        'investor_num_companies',
        'log_investor_avg_investment',
        'log_investor_days_since_last',
        
        'category_overlap',
        'common_co_investors',
        'log_investor_experience'
    ]
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 10 des features les plus importantes:")
    for i in range(min(10, len(feature_names))):
        print(f"  {i+1:2d}. {feature_names[indices[i]]:<35} {importances[indices[i]]:.4f}")
    
    # 10. Résumé final
    print("\n" + "="*80)
    print("RÉSUMÉ FINAL - COMPARAISON AVEC TGN")
    print("="*80)

    print(f"""
    RANDOM FOREST BASELINE (toutes métriques):

    VALIDATION SET:
      Classification Metrics:
         - AUROC:             {val_auc:.4f}
         - Average Precision: {val_ap:.4f}

      Ranking Metrics:
         - MRR:               {val_mrr:.4f}
         - Recall@10:         {val_recall_k[10]:.4f}
         - Recall@50:         {val_recall_k[50]:.4f}

    TEST SET:
      Classification Metrics:
         - AUROC:             {test_auc:.4f}
         - Average Precision: {test_ap:.4f}

      Ranking Metrics:
         - MRR:               {test_mrr:.4f}
         - Recall@10:         {test_recall_k[10]:.4f}
         - Recall@50:         {test_recall_k[50]:.4f}

    📋 INTERPRÉTATION:
       - Ces scores représentent la baseline pour votre TGN
       - TGN devrait améliorer ces scores grâce à la capture de la dynamique temporelle
       - Si TGN < RF: problème dans l'implémentation TGN
       - Si TGN ≈ RF: TGN ne capture pas mieux la temporalité
       - Si TGN > RF (+5-15%): TGN fonctionne bien!

    OBJECTIFS POUR TGN (TEST SET):
       - AUROC:     > {test_auc + 0.05:.4f} (+5%)
       - AP:        > {test_ap + 0.05:.4f} (+5%)
       - MRR:       > {test_mrr + 0.05:.4f} (+5%)
       - Recall@10: > {test_recall_k[10] + 0.05:.4f} (+5%)
       - Recall@50: > {test_recall_k[50] + 0.05:.4f} (+5%)

    [OK] IMPORTANT:
       - Split temporel 70/15/15 (train/val/test) comme TGN
       - Même évaluation temporelle que TGN (prédiction FUTURE uniquement)
       - Pas de data leakage (features calculées sur historique uniquement)
       - 5 métriques identiques pour comparaison directe
    """)
    
    # 11. Sauvegarde des résultats
    results = {
        'validation': {
            'auroc': val_auc,
            'ap': val_ap,
            'mrr': val_mrr,
            'recall@10': val_recall_k[10],
            'recall@50': val_recall_k[50]
        },
        'test': {
            'auroc': test_auc,
            'ap': test_ap,
            'mrr': test_mrr,
            'recall@10': test_recall_k[10],
            'recall@50': test_recall_k[50]
        },
        'feature_importances': dict(zip(feature_names, importances))
    }

    import json
    with open('rf_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n[OK] Résultats sauvegardés dans: rf_baseline_results.json")

if __name__ == "__main__":
    main()