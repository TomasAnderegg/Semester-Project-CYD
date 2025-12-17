"""
Test du mécanisme de ranking (version Python pur, sans PyTorch)
Pour comprendre comment on détecte si Top Proba = Vrai Prédiction
"""

import numpy as np

def find_rank_of_true_investor(all_scores):
    """
    Trouve le rang du vrai investisseur (toujours à l'indice 0).

    Args:
        all_scores: Liste de scores [vrai_score, neg1, neg2, ...]

    Returns:
        rang du vrai investisseur (1 = meilleur)
    """
    # Trier les indices par score décroissant
    sorted_indices = np.argsort(all_scores)[::-1]  # [::-1] pour descending

    # Trouver la position de l'indice 0 (vrai investisseur)
    rank = np.where(sorted_indices == 0)[0][0] + 1  # +1 car rang commence à 1

    return rank, sorted_indices


print("=" * 80)
print("TEST: Comment on détecte si Top Proba = Vrai Prédiction")
print("=" * 80)

# ============================================================
# CAS 1: Top proba = vrai investisseur (PARFAIT)
# ============================================================
print("\n[CAS 1] Top proba = Vrai investisseur (Sequoia)")
print("-" * 80)

scores = [0.78, 0.65, 0.23, 0.19, 0.12]
#         ↑     ↑     ↑     ↑     ↑
#      Sequoia Accel a16z  YComb Other
#       VRAI   négatifs...

print("Scores par candidat:")
print(f"  Index 0 (Sequoia, VRAI): {scores[0]:.2f}")
print(f"  Index 1 (Accel):         {scores[1]:.2f}")
print(f"  Index 2 (a16z):          {scores[2]:.2f}")
print(f"  Index 3 (Y Comb):        {scores[3]:.2f}")
print(f"  Index 4 (Other):         {scores[4]:.2f}")

rank, sorted_indices = find_rank_of_true_investor(np.array(scores))

print(f"\nApres tri descending:")
print(f"  Classement: {sorted_indices}")
print(f"  -> Indice {sorted_indices[0]} a le score le plus eleve ({scores[sorted_indices[0]]:.2f})")

print(f"\nPosition du vrai investisseur (indice 0): RANG {rank}")

if rank == 1:
    print("[OK] TOP PROBA = VRAI PREDICTION!")
    print(f"   MRR contribution: 1/{rank} = {1.0/rank:.2f}")
    print(f"   Recall@1: True [OK]")
    print(f"   Recall@10: True [OK]")
else:
    print(f"[X] Top proba != vrai prediction")

# ============================================================
# CAS 2: 3ème proba = vrai investisseur (MOYEN)
# ============================================================
print("\n\n[CAS 2] 3ème proba = Vrai investisseur (Sequoia)")
print("-" * 80)

scores = [0.35, 0.78, 0.65, 0.23, 0.19]
#         ↑     ↑     ↑     ↑     ↑
#      Sequoia Accel a16z  YComb Other
#       VRAI   (ce négatif a un meilleur score!)

print("Scores par candidat:")
for i, score in enumerate(scores):
    label = "VRAI" if i == 0 else "négatif"
    print(f"  Index {i} ({label}): {score:.2f}")

rank, sorted_indices = find_rank_of_true_investor(np.array(scores))

print(f"\nApres tri descending:")
print(f"  Classement: {sorted_indices}")
print(f"  Top 3: [{sorted_indices[0]}, {sorted_indices[1]}, {sorted_indices[2]}]")

print(f"\nPosition du vrai investisseur (indice 0): RANG {rank}")

if rank != 1:
    print(f"[X] Top proba != vrai prediction")
    print(f"   Top proba = indice {sorted_indices[0]} (score {scores[sorted_indices[0]]:.2f})")
    print(f"   MRR contribution: 1/{rank} = {1.0/rank:.2f}")
    print(f"   Recall@1: False [X]")
    print(f"   Recall@3: {rank <= 3} {'[OK]' if rank <= 3 else '[X]'}")
    print(f"   Recall@10: {rank <= 10} {'[OK]' if rank <= 10 else '[X]'}")

# ============================================================
# CAS 3: 50ème proba = vrai investisseur (MAUVAIS)
# ============================================================
print("\n\n[CAS 3] 50ème proba = Vrai investisseur (Sequoia)")
print("-" * 80)

# Vrai investisseur a un mauvais score, 100 négatifs avec des meilleurs scores
scores = [0.05] + list(np.linspace(0.95, 0.06, 100))

print(f"Score vrai investisseur (indice 0): {scores[0]:.3f}")
print(f"Score top négatif (indice 1): {scores[1]:.3f}")
print(f"Score 10ème négatif (indice 10): {scores[10]:.3f}")
print(f"Score 50ème négatif (indice 50): {scores[50]:.3f}")

rank, sorted_indices = find_rank_of_true_investor(np.array(scores))

print(f"\nTop 5 du classement: {sorted_indices[:5]}")
print(f"Position du vrai investisseur (indice 0): RANG {rank}")

print(f"[X] Tres mauvaise prediction!")
print(f"   MRR contribution: 1/{rank} = {1.0/rank:.4f}")
print(f"   Recall@1: False [X]")
print(f"   Recall@10: False [X]")
print(f"   Recall@50: {rank <= 50} {'[OK]' if rank <= 50 else '[X]'}")

# ============================================================
# BATCH EXAMPLE: Calculer MRR et Recall sur 3 startups
# ============================================================
print("\n\n" + "=" * 80)
print("BATCH EXAMPLE: 3 startups ensemble")
print("=" * 80)

startups = [
    ("QuantumTech", [0.78, 0.65, 0.23, 0.19]),  # Vrai investisseur classé 1er
    ("BioStartup", [0.35, 0.78, 0.65, 0.52]),   # Vrai investisseur classé 3ème
    ("AICompany", [0.05, 0.95, 0.85, 0.75]),    # Vrai investisseur classé 4ème (dernier)
]

ranks = []
for name, scores in startups:
    rank, sorted_indices = find_rank_of_true_investor(np.array(scores))
    ranks.append(rank)
    print(f"\n{name}:")
    print(f"  Scores: {scores}")
    print(f"  Classement: {sorted_indices}")
    print(f"  Rang du vrai investisseur: {rank}")

print(f"\n{'-' * 80}")
print(f"Rangs des vrais investisseurs: {ranks}")

# MRR
reciprocal_ranks = [1.0/r for r in ranks]
mrr = np.mean(reciprocal_ranks)
print(f"\nMRR = moyenne(1/rang)")
print(f"    = moyenne([1/{ranks[0]}, 1/{ranks[1]}, 1/{ranks[2]}])")
print(f"    = moyenne({reciprocal_ranks})")
print(f"    = {mrr:.3f}")

# Recall@K
for k in [1, 3, 10]:
    in_top_k = [r <= k for r in ranks]
    recall = np.mean(in_top_k)
    count = sum(in_top_k)
    print(f"Recall@{k} = {count}/{len(ranks)} = {recall:.2%}")

print("\n" + "=" * 80)
print("RESUME DE L'IMPLEMENTATION:")
print("=" * 80)
print("""
1. CONVENTION: Le vrai investisseur est TOUJOURS a l'indice 0
   all_scores = [vrai_score, neg1, neg2, ..., neg100]
                 ^ Indice 0

2. TRI DESCENDING: Trier tous les scores du plus haut au plus bas
   sorted_indices = argsort(all_scores, descending=True)

3. TROUVER POSITION: Chercher ou se trouve l'indice 0 dans le classement
   position = where(sorted_indices == 0)
   rank = position + 1  (car rangs commencent a 1)

4. VERIFICATION TOP PROBA:
   [OK] Si rank == 1 -> Top proba = Vrai prediction
   [X] Si rank > 1  -> Top proba != Vrai prediction

5. METRIQUES:
   - MRR contribution = 1/rank
   - Recall@K = True si rank <= K, False sinon
""")
print("=" * 80)
