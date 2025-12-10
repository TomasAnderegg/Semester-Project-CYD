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

# Chemin vers votre CSV brut avec toutes les transactions
RAW_DATA_PATH = "debug_df_graphcaca21.csv"  # Le CSV avec toutes les données

def load_data():
    """Charge toutes les données nécessaires"""
    print("📁 Chargement des données...")
    
    # 1. Données TGN (edges positives seulement)
    tgn_edges = pd.read_csv(TGN_EDGES_PATH)
    print(f"  Edges TGN chargées: {len(tgn_edges):,}")
    
    # 2. Données brutes avec toutes les transactions
    raw_data = pd.read_csv(RAW_DATA_PATH)
    print(f"  Données brutes chargées: {len(raw_data):,}")
    print(f"  Colonnes: {raw_data.columns.tolist()}")
    
    # 3. Mappings
    with open(COMPANY_MAP_PATH, 'rb') as f:
        company_map = pickle.load(f)
    
    with open(INVESTOR_MAP_PATH, 'rb') as f:
        investor_map = pickle.load(f)
    
    # Inverser les mappings
    id_to_company = {v: k for k, v in company_map.items()}
    id_to_investor = {v: k for k, v in investor_map.items()}
    
    return tgn_edges, raw_data, id_to_company, id_to_investor

def create_temporal_splits(tgn_edges):
    """Crée des splits temporels pour prédiction FUTURE"""
    # Trier par timestamp
    tgn_edges = tgn_edges.sort_values('ts')
    
    # Split temporel (70/15/15)
    train_cutoff = tgn_edges['ts'].quantile(0.70)
    val_cutoff = tgn_edges['ts'].quantile(0.85)
    
    print(f"\n📅 Split temporel:")
    print(f"  Train: ts ≤ {train_cutoff}")
    print(f"  Val:   {train_cutoff} < ts ≤ {val_cutoff}")
    print(f"  Test:  ts > {val_cutoff}")
    
    # Convertir en datetime pour faciliter
    tgn_edges['datetime'] = pd.to_datetime(tgn_edges['ts'], unit='s')
    
    return train_cutoff, val_cutoff, tgn_edges

def compute_company_features(raw_data, company, timestamp):
    """Calcule les features d'une compagnie JUSQU'À timestamp (pas après)"""
    # Filtrer les transactions de cette compagnie AVANT timestamp
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
            'days_since_last_round': 365*5  # 5 ans par défaut
        }
    
    # Features
    total_raised = company_data['raised_amount_usd'].sum()
    num_rounds = len(company_data)
    num_investors = company_data['investor_name'].nunique()
    avg_round_size = total_raised / num_rounds if num_rounds > 0 else 0
    
    # Temps depuis le dernier round
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
    # Filtrer les transactions de cet investisseur AVANT timestamp
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
            'days_since_last_investment': 365*2  # 2 ans par défaut
        }
    
    # Features
    total_invested = investor_data['raised_amount_usd'].sum()
    num_investments = len(investor_data)
    num_companies = investor_data['org_name'].nunique()
    avg_investment_size = total_invested / num_investments if num_investments > 0 else 0
    
    # Temps depuis le dernier investissement
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
    # 1. Historique de l'investisseur avec des compagnies similaires
    investor_history = raw_data[
        (raw_data['investor_name'] == investor) & 
        (pd.to_datetime(raw_data['announced_on']) <= pd.to_datetime(timestamp, unit='s'))
    ]
    
    # 2. Catégories préférées de l'investisseur
    investor_companies = investor_history['org_name'].unique()
    investor_categories = set()
    
    for comp in investor_companies:
        comp_data = raw_data[raw_data['org_name'] == comp]
        if not comp_data.empty and 'category_list' in comp_data.columns:
            cats = comp_data['category_list'].iloc[0]
            if isinstance(cats, str):
                investor_categories.update([c.strip() for c in cats.split(',')])
    
    # 3. Catégories de la compagnie
    company_data = raw_data[raw_data['org_name'] == company]
    company_categories = set()
    if not company_data.empty and 'category_list' in company_data.columns:
        cats = company_data['category_list'].iloc[0]
        if isinstance(cats, str):
            company_categories.update([c.strip() for c in cats.split(',')])
    
    # 4. Similarité de catégories
    if investor_categories and company_categories:
        category_overlap = len(investor_categories & company_categories) / len(investor_categories | company_categories)
    else:
        category_overlap = 0
    
    # 5. Co-investisseurs communs (réseau)
    company_investors = set(raw_data[
        (raw_data['org_name'] == company) & 
        (pd.to_datetime(raw_data['announced_on']) <= pd.to_datetime(timestamp, unit='s'))
    ]['investor_name'].unique())
    
    common_co_investors = 0
    for other_inv in company_investors:
        if other_inv == investor:
            continue
        # Vérifier si cet autre investisseur a investi dans les mêmes compagnies que notre investisseur
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
        'investor_experience_with_similar': len(investor_companies)  # Nombre total d'investissements
    }

def create_positive_samples(tgn_edges, raw_data, id_to_company, id_to_investor, split_type='test'):
    """Crée des échantillons POSITIFS pour la prédiction FUTURE"""
    positives = []
    
    # Filtrer selon le split
    if split_type == 'train':
        edges = tgn_edges[tgn_edges['ts'] <= train_cutoff]
    elif split_type == 'val':
        edges = tgn_edges[(tgn_edges['ts'] > train_cutoff) & (tgn_edges['ts'] <= val_cutoff)]
    else:  # test
        edges = tgn_edges[tgn_edges['ts'] > val_cutoff]
    
    print(f"\n🔍 Création des positifs ({split_type}): {len(edges)} edges")
    
    for idx, row in tqdm(edges.iterrows(), total=len(edges), desc=f"Processing {split_type} positives"):
        company = id_to_company.get(int(row['u']), f"company_{row['u']}")
        investor = id_to_investor.get(int(row['i']), f"investor_{row['i']}")
        timestamp = row['ts']
        
        # Features de la compagnie (AVANT timestamp)
        company_feats = compute_company_features(raw_data, company, timestamp)
        
        # Features de l'investisseur (AVANT timestamp)
        investor_feats = compute_investor_features(raw_data, investor, timestamp)
        
        # Features de la paire (AVANT timestamp)
        pair_feats = compute_pair_features(raw_data, company, investor, timestamp)
        
        # Combiner toutes les features
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
        
        positives.append({
            'company': company,
            'investor': investor,
            'timestamp': timestamp,
            'features': features,
            'label': 1
        })
    
    return positives

def create_negative_samples(positives, raw_data, num_negatives_per_positive=1):
    """Crée des échantillons NÉGATIFS plausibles"""
    negatives = []
    
    # Liste de toutes les compagnies et investisseurs
    all_companies = raw_data['org_name'].unique()
    all_investors = raw_data['investor_name'].unique()
    
    print(f"\n🔍 Création des négatifs: {len(positives)} × {num_negatives_per_positive}")
    
    for pos in tqdm(positives, desc="Generating negatives"):
        company = pos['company']
        investor = pos['investor']
        timestamp = pos['timestamp']
        
        # Trouver des compagnies que cet investisseur n'a PAS financées avant ce timestamp
        funded_companies = set(raw_data[
            (raw_data['investor_name'] == investor) & 
            (pd.to_datetime(raw_data['announced_on']) <= pd.to_datetime(timestamp, unit='s'))
        ]['org_name'].unique())
        
        # Candidates: compagnies non financées par cet investisseur
        candidate_companies = [c for c in all_companies 
                              if c not in funded_companies and c != company]
        
        # Prendre quelques négatifs aléatoires
        num_to_sample = min(num_negatives_per_positive, len(candidate_companies))
        if num_to_sample > 0:
            negative_companies = np.random.choice(candidate_companies, size=num_to_sample, replace=False)
            
            for neg_company in negative_companies:
                # Mêmes features que pour les positifs
                company_feats = compute_company_features(raw_data, neg_company, timestamp)
                investor_feats = compute_investor_features(raw_data, investor, timestamp)
                pair_feats = compute_pair_features(raw_data, neg_company, investor, timestamp)
                
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
                
                negatives.append({
                    'company': neg_company,
                    'investor': investor,
                    'timestamp': timestamp,
                    'features': features,
                    'label': 0
                })
    
    return negatives

def prepare_dataset(positives, negatives):
    """Prépare les matrices X et y pour l'entraînement"""
    all_samples = positives + negatives
    np.random.shuffle(all_samples)
    
    X = np.array([s['features'] for s in all_samples])
    y = np.array([s['label'] for s in all_samples])
    
    print(f"\n📊 Dataset final:")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Positives: {len(positives)} ({100*len(positives)/len(all_samples):.1f}%)")
    print(f"  Negatives: {len(negatives)} ({100*len(negatives)/len(all_samples):.1f}%)")
    print(f"  Features: {X.shape[1]}")
    
    return X, y

def main():
    print("="*70)
    print("RANDOM FOREST - Prédiction de liens FUTURS (sans leakage)")
    print("="*70)
    
    # 1. Charger les données
    tgn_edges, raw_data, id_to_company, id_to_investor = load_data()
    
    # 2. Créer les splits temporels
    global train_cutoff, val_cutoff
    train_cutoff, val_cutoff, tgn_edges = create_temporal_splits(tgn_edges)
    
    # 3. Préparer les données brutes
    raw_data['announced_on'] = pd.to_datetime(raw_data['announced_on'], errors='coerce')
    raw_data = raw_data.dropna(subset=['announced_on', 'org_name', 'investor_name'])
    
    # 4. Créer les datasets
    print("\n" + "="*70)
    print("CRÉATION DES DATASETS")
    print("="*70)
    
    # Train
    train_positives = create_positive_samples(tgn_edges, raw_data, id_to_company, id_to_investor, 'train')
    train_negatives = create_negative_samples(train_positives, raw_data, num_negatives_per_positive=1)
    X_train, y_train = prepare_dataset(train_positives, train_negatives)
    
    # Test
    test_positives = create_positive_samples(tgn_edges, raw_data, id_to_company, id_to_investor, 'test')
    test_negatives = create_negative_samples(test_positives, raw_data, num_negatives_per_positive=1)
    X_test, y_test = prepare_dataset(test_positives, test_negatives)
    
    # 5. Entraînement
    print("\n" + "="*70)
    print("ENTRAÎNEMENT DU RANDOM FOREST")
    print("="*70)
    
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
    
    print("\n🔧 Entraînement en cours...")
    rf.fit(X_train, y_train)
    
    # 6. Évaluation
    print("\n" + "="*70)
    print("ÉVALUATION")
    print("="*70)
    
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    
    print(f"📊 Résultats:")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    
    # 7. Analyse
    print("\n" + "="*70)
    print("ANALYSE")
    print("="*70)
    
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
    
    print("\n📈 Top 10 des features les plus importantes:")
    for i in range(min(10, len(feature_names))):
        print(f"  {i+1:2d}. {feature_names[indices[i]]:<30} {importances[indices[i]]:.4f}")
    
    # 8. Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ POUR COMPARAISON AVEC TGN")
    print("="*70)
    
    print(f"""
    🎯 RANDOM FOREST (prédiction FUTURE, sans leakage):
       - Test AUC: {auc:.4f}
       - Test AP:  {ap:.4f}
       - {len(feature_names)} features temporelles/historiques
    
    🔍 INTERPRÉTATION:
       - AUC attendue: 0.65-0.80 (réaliste)
       - Si AUC < 0.6: Tâche très difficile
       - Si AUC > 0.85: Vérifier d'éventuels leakages
    
    🎯 POUR TGN:
       - Objectif: AUC > {auc + 0.05:.4f} (+5%)
       - Si TGN a AUC < {auc:.4f}: Il sous-performe le RF simple
       - Si TGN a AUC > {auc + 0.10:.4f}: Il capture bien la dynamique temporelle
    
    ✅ CE QUI A CHANGÉ:
       1. Features calculées UNIQUEMENT sur l'historique (pas de futur)
       2. Mêmes features pour positifs ET négatifs
       3. Prédiction de liens FUTURS (pas reconnaissance de liens existants)
       4. Pas de features spécifiques à la paire (comme total_raised entre C et I)
    """)

if __name__ == "__main__":
    main()