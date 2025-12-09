import numpy as np
import pandas as pd
import pickle

# ==============================
# VÉRIFICATION DES FEATURES TGN
# ==============================

def verify_tgn_data():
    """
    Vérifie ce que le TGN reçoit vraiment comme features.
    """
    print("\n" + "="*70)
    print("VÉRIFICATION DES FEATURES TGN")
    print("="*70)
    
    # Charger les données
    df = pd.read_csv("data/crunchbase_filtered.csv")
    edge_feats = np.load("data/crunchbase_filtered.npy")
    node_feats = np.load("data/crunchbase_filtered_node.npy")
    
    print(f"\n📊 DATASET TGN:")
    print(f"  Nombre d'edges : {len(df)}")
    print(f"  Nombre de labels=1 : {(df['label']==1).sum()}")
    print(f"  Nombre de labels=0 : {(df['label']==0).sum()}")
    print(f"  Ratio pos:neg : {(df['label']==1).sum()}:{(df['label']==0).sum()}")
    
    # Node features
    print(f"\n📊 NODE FEATURES:")
    print(f"  Shape: {node_feats.shape}")
    print(f"  Non-zero entries: {(node_feats != 0).sum()}/{node_feats.size}")
    print(f"  Mean: {node_feats.mean():.6f}")
    print(f"  Max: {node_feats.max():.6f}")
    
    if (node_feats == 0).all():
        print("  ❌ PROBLÈME: Toutes les node features sont à zéro!")
        print("     Le TGN ne voit PAS les degrés comme le RF")
    
    # Edge features
    print(f"\n📊 EDGE FEATURES:")
    print(f"  Shape: {edge_feats.shape}")
    print(f"  Feature 0 (log raised):")
    print(f"    Mean: {edge_feats[:, 0].mean():.2f}")
    print(f"    Non-zero: {(edge_feats[:, 0] > 0).sum()}/{len(edge_feats)} ({100*(edge_feats[:, 0] > 0).mean():.1f}%)")
    print(f"  Feature 1 (num_rounds):")
    print(f"    Mean: {edge_feats[:, 1].mean():.2f}")
    print(f"    Non-zero: {(edge_feats[:, 1] > 0).sum()}/{len(edge_feats)} ({100*(edge_feats[:, 1] > 0).mean():.1f}%)")
    
    # CRITIQUE: Corrélation avec labels
    print(f"\n🔍 CORRÉLATION AVEC LABELS:")
    print(f"  Feature 0 vs label: {np.corrcoef(edge_feats[:, 0], df['label'])[0, 1]:.4f}")
    print(f"  Feature 1 vs label: {np.corrcoef(edge_feats[:, 1], df['label'])[0, 1]:.4f}")
    
    if abs(np.corrcoef(edge_feats[:, 0], df['label'])[0, 1]) > 0.9:
        print("  ❌ LEAKAGE DÉTECTÉ: Corrélation très forte!")
        print("     Les edge features révèlent directement les labels")
    
    # Distribution par label
    pos_idx = df['label'] == 1
    neg_idx = df['label'] == 0
    
    print(f"\n📊 DISTRIBUTION PAR LABEL:")
    print(f"  POSITIVES (label=1):")
    print(f"    Feature 0 - mean: {edge_feats[pos_idx, 0].mean():.2f}, non-zero: {(edge_feats[pos_idx, 0] > 0).mean()*100:.1f}%")
    print(f"    Feature 1 - mean: {edge_feats[pos_idx, 1].mean():.2f}, non-zero: {(edge_feats[pos_idx, 1] > 0).mean()*100:.1f}%")
    
    print(f"  NÉGATIVES (label=0):")
    print(f"    Feature 0 - mean: {edge_feats[neg_idx, 0].mean():.2f}, non-zero: {(edge_feats[neg_idx, 0] > 0).mean()*100:.1f}%")
    print(f"    Feature 1 - mean: {edge_feats[neg_idx, 1].mean():.2f}, non-zero: {(edge_feats[neg_idx, 1] > 0).mean()*100:.1f}%")
    
    # Diagnostic
    print(f"\n" + "="*70)
    print("DIAGNOSTIC")
    print("="*70)
    
    neg_all_zero = (edge_feats[neg_idx] == 0).all()
    if neg_all_zero:
        print("❌ PROBLÈME MAJEUR:")
        print("   Toutes les négatives ont edge_features = [0, 0]")
        print("   Le TGN devrait facilement apprendre: if feat==[0,0]: predict 0")
        print("   → Il devrait aussi obtenir AUC ≈ 1.0")
    else:
        print("✅ Les négatives ont des edge features variées")
        print("   Pas de leakage évident dans les edge features")
    
    pos_all_nonzero = (edge_feats[pos_idx] > 0).all()
    if pos_all_nonzero:
        print("\n❌ PROBLÈME:")
        print("   Toutes les positives ont edge_features > 0")
        print("   Séparation parfaite possible")
    
    # Recommandations
    print(f"\n" + "="*70)
    print("RECOMMANDATIONS")
    print("="*70)
    
    if (node_feats == 0).all():
        print("\n1. NODE FEATURES VIDES:")
        print("   Le TGN n'a pas accès aux degrés comme le RF")
        print("   → Enrichir les node features avec degrés, activité, etc.")
        print("   → Voir l'artifact 'fix_node_features' précédent")
    
    if neg_all_zero:
        print("\n2. EDGE FEATURES LEAKAGE:")
        print("   Les négatives ont TOUJOURS [0, 0]")
        print("   → Retirer raised/rounds des edge features")
        print("   → Ou générer des négatives plus réalistes")
    
    print("\n3. COMPARAISON ÉQUITABLE:")
    print("   Pour comparer avec RF, deux options:")
    print("   a) RF SANS raised/rounds (degrés uniquement) vs TGN actuel")
    print("   b) Enrichir TGN avec node features riches, puis comparer")


def check_negative_sampling():
    """
    Vérifie comment les négatives sont générées dans TGN.
    """
    print("\n" + "="*70)
    print("VÉRIFICATION NEGATIVE SAMPLING")
    print("="*70)
    
    df = pd.read_csv("data/crunchbase_filtered.csv")
    
    # Le CSV TGN contient UNIQUEMENT les positives
    # Les négatives sont générées à la volée pendant training
    print(f"\nCSV TGN:")
    print(f"  Total edges: {len(df)}")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")
    
    if (df['label'] == 1).all():
        print("\n✅ CORRECT: Le CSV contient UNIQUEMENT les positives")
        print("   Les négatives sont générées dynamiquement pendant training")
        print("   → Elles n'ont PAS d'edge features pré-calculées")
        print("   → TGN utilise probablement des features par défaut [0, 0]")
        
        print("\n💡 EXPLICATION:")
        print("   Dans train_supervised.py:")
        print("   - _, negatives_batch = train_rand_sampler.sample(size)")
        print("   - Ces négatives sont juste des IDs d'investisseurs")
        print("   - Quand TGN calcule neg_prob, il n'a pas de 'edge_idx'")
        print("   - Donc il utilise probablement edge_features = [0, 0]")
        
        print("\n🎯 CONCLUSION:")
        print("   Le TGN a le MÊME problème que le RF initial:")
        print("   - Positives: edge_features ≠ 0")
        print("   - Négatives: edge_features = [0, 0]")
        print("   → Séparation parfaite possible")
    else:
        print(f"\n⚠️ Le CSV contient déjà des négatives!")
        print("   C'est inhabituel pour TGN")


if __name__ == "__main__":
    verify_tgn_data()
    check_negative_sampling()