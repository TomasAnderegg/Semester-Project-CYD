import pandas as pd
import numpy as np

# Charger vos données
df = pd.read_csv("data/crunchbase_tgn.csv")

print("=" * 70)
print("ANALYSE DES TIMESTAMPS")
print("=" * 70)

# 1. Vérifier si les données sont bien triées
print("\n1️⃣ Vérification du tri global:")
is_sorted = (df.ts.diff().dropna() >= 0).all()
print(f"   Données triées: {is_sorted}")
if not is_sorted:
    print("   [ERROR] PROBLÈME: Les timestamps ne sont pas triés!")
    bad_indices = df[df.ts.diff() < 0].index
    print(f"   Indices problématiques: {bad_indices.tolist()[:10]}")

# 2. Vérifier les doublons de timestamps
print("\n2️⃣ Doublons de timestamps:")
duplicate_ts = df[df.ts.duplicated(keep=False)]
print(f"   Nombre de timestamps dupliqués: {len(duplicate_ts)}")
if len(duplicate_ts) > 0:
    print(f"   Exemple: {duplicate_ts.head(10)}")

# 3. Vérifier les interactions par nœud
print("\n3️⃣ Analyse par nœud (source):")
node_interactions = df.groupby('u').agg({
    'ts': ['count', 'min', 'max', lambda x: (x.diff().dropna() < 0).any()]
}).round(2)
node_interactions.columns = ['count', 'min_ts', 'max_ts', 'has_backward_time']

problematic_nodes = node_interactions[node_interactions['has_backward_time'] == True]
print(f"   Nœuds sources avec timestamps inversés: {len(problematic_nodes)}")
if len(problematic_nodes) > 0:
    print(f"   Exemples:\n{problematic_nodes.head()}")

print("\n4️⃣ Analyse par nœud (destination):")
node_interactions_dst = df.groupby('i').agg({
    'ts': ['count', 'min', 'max', lambda x: (x.diff().dropna() < 0).any()]
}).round(2)
node_interactions_dst.columns = ['count', 'min_ts', 'max_ts', 'has_backward_time']

problematic_nodes_dst = node_interactions_dst[node_interactions_dst['has_backward_time'] == True]
print(f"   Nœuds destinations avec timestamps inversés: {len(problematic_nodes_dst)}")
if len(problematic_nodes_dst) > 0:
    print(f"   Exemples:\n{problematic_nodes_dst.head()}")

# 5. Simuler le problème batch
print("\n5️⃣ Simulation du problème dans un batch:")
BATCH_SIZE = 200
start_idx = 0
end_idx = min(len(df), start_idx + BATCH_SIZE)
batch = df.iloc[start_idx:end_idx]

print(f"   Batch de {start_idx} à {end_idx}")
print(f"   Timestamps uniques dans le batch: {batch.ts.nunique()}")
print(f"   Nœuds sources uniques: {batch.u.nunique()}")
print(f"   Nœuds destinations uniques: {batch.i.nunique()}")

# Vérifier si un même nœud apparaît plusieurs fois avec des timestamps différents
for node_col, node_type in [('u', 'source'), ('i', 'destination')]:
    duplicated_nodes = batch[batch[node_col].duplicated(keep=False)]
    if len(duplicated_nodes) > 0:
        print(f"\n   [WARNING]  Nœuds {node_type} apparaissant plusieurs fois dans le batch:")
        for node in duplicated_nodes[node_col].unique()[:5]:
            node_data = duplicated_nodes[duplicated_nodes[node_col] == node][['u', 'i', 'ts']].sort_values('ts')
            print(f"      Node {node}:")
            print(f"{node_data.to_string(index=False)}")
            if len(node_data) > 1:
                ts_diff = node_data.ts.diff().dropna()
                if (ts_diff < 0).any():
                    print(f"         [ERROR] TIMESTAMPS INVERSÉS!")

# 6. Vérifier les timestamps = 0
print("\n6️⃣ Timestamps à zéro:")
zero_ts = df[df.ts == 0]
print(f"   Nombre: {len(zero_ts)}")
if len(zero_ts) > 0:
    print(f"   Pourcentage: {len(zero_ts)/len(df)*100:.2f}%")
    print(f"   [WARNING]  Ces interactions seront toutes au même timestamp!")

print("\n" + "=" * 70)
print("RECOMMANDATIONS")
print("=" * 70)

if not is_sorted:
    print("[ERROR] Les données ne sont PAS triées par timestamp")
    print("   → Ajoutez df.sort_values('ts') dans prepare_tgn_input()")
elif len(zero_ts) > len(df) * 0.1:
    print("[WARNING]  Plus de 10% des timestamps sont à 0")
    print("   → Filtrez ces lignes OU imputez des timestamps artificiels")
elif len(problematic_nodes) > 0 or len(problematic_nodes_dst) > 0:
    print("[WARNING]  Des nœuds ont des interactions non ordonnées")
    print("   → Le tri global ne suffit pas, il faut gérer les timestamps égaux")
else:
    print("[OK] Pas de problème évident détecté")
    print("   Le problème vient probablement du code TGN lui-même")