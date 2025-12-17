"""
Test du mécanisme de ranking pour vérifier comment on détecte
si la top probabilité correspond à la vraie prédiction.
"""

import torch

def test_ranking_mechanism():
    """
    Simule 3 cas:
    1. Top proba = vrai investisseur (parfait)
    2. 5ème proba = vrai investisseur (moyen)
    3. 50ème proba = vrai investisseur (mauvais)
    """

    print("=" * 70)
    print("TEST: Comment détecter si Top Proba = Vrai Prédiction")
    print("=" * 70)

    # ============================================================
    # CAS 1: Top proba = vrai investisseur (PARFAIT)
    # ============================================================
    print("\n[CAS 1] Top proba = Vrai investisseur (Sequoia)")
    print("-" * 70)

    # Scores: vrai investisseur (indice 0) a le PLUS HAUT score
    pos_score = torch.tensor([[0.78]])  # Sequoia (vrai)
    neg_scores = torch.tensor([[0.65, 0.23, 0.19, 0.12, 0.08]])  # 5 négatifs

    # Combine (vrai investisseur TOUJOURS à l'indice 0)
    all_scores = torch.cat([pos_score, neg_scores], dim=1)
    print(f"Scores: {all_scores.numpy()}")
    print(f"  Index 0 (Sequoia, VRAI): {all_scores[0, 0].item():.2f}")
    print(f"  Index 1-5 (négatifs):     {neg_scores.numpy()}")

    # Classement descending
    rankings = torch.argsort(all_scores, dim=1, descending=True)
    print(f"\nClassement (indices triés): {rankings.numpy()}")
    print(f"  → Indice {rankings[0, 0].item()} a le score le plus élevé")

    # Trouver position du vrai (indice 0)
    positive_rank = (rankings == 0).nonzero(as_tuple=True)[1] + 1
    print(f"\nPosition du vrai investisseur (indice 0): RANG {positive_rank.item()}")

    # Vérification
    if positive_rank == 1:
        print("✅ TOP PROBA = VRAI PRÉDICTION!")
        print(f"   MRR contribution: 1/{positive_rank.item()} = {1.0/positive_rank.item():.2f}")
        print(f"   Recall@1: True")
        print(f"   Recall@10: True")

    # ============================================================
    # CAS 2: 5ème proba = vrai investisseur (MOYEN)
    # ============================================================
    print("\n\n[CAS 2] 5ème proba = Vrai investisseur (Sequoia)")
    print("-" * 70)

    # Scores: vrai investisseur a le 5ème score
    pos_score = torch.tensor([[0.35]])  # Sequoia (vrai)
    neg_scores = torch.tensor([[0.78, 0.65, 0.52, 0.41, 0.23, 0.19]])  # 6 négatifs

    all_scores = torch.cat([pos_score, neg_scores], dim=1)
    print(f"Scores: {all_scores.numpy()}")
    print(f"  Index 0 (Sequoia, VRAI): {all_scores[0, 0].item():.2f}")

    rankings = torch.argsort(all_scores, dim=1, descending=True)
    print(f"\nClassement (indices triés): {rankings.numpy()}")

    positive_rank = (rankings == 0).nonzero(as_tuple=True)[1] + 1
    print(f"\nPosition du vrai investisseur (indice 0): RANG {positive_rank.item()}")

    if positive_rank != 1:
        print(f"❌ Top proba ≠ vrai prédiction")
        print(f"   Top proba = indice {rankings[0, 0].item()} (score {all_scores[0, rankings[0, 0]].item():.2f})")
        print(f"   MRR contribution: 1/{positive_rank.item()} = {1.0/positive_rank.item():.2f}")
        print(f"   Recall@1: False")
        print(f"   Recall@10: {positive_rank.item() <= 10}")

    # ============================================================
    # CAS 3: 50ème proba = vrai investisseur (MAUVAIS)
    # ============================================================
    print("\n\n[CAS 3] 50ème proba = Vrai investisseur (Sequoia)")
    print("-" * 70)

    # Scores: vrai investisseur a un très mauvais score
    pos_score = torch.tensor([[0.05]])  # Sequoia (vrai)
    # 100 négatifs avec des scores entre 0.06 et 0.95
    neg_scores = torch.linspace(0.95, 0.06, 100).unsqueeze(0)

    all_scores = torch.cat([pos_score, neg_scores], dim=1)
    print(f"Score vrai investisseur (indice 0): {all_scores[0, 0].item():.3f}")
    print(f"Score top négatif (indice 1): {all_scores[0, 1].item():.3f}")
    print(f"Score 50ème négatif (indice 50): {all_scores[0, 50].item():.3f}")

    rankings = torch.argsort(all_scores, dim=1, descending=True)
    print(f"\nTop 3 du classement: {rankings[0, :3].numpy()}")

    positive_rank = (rankings == 0).nonzero(as_tuple=True)[1] + 1
    print(f"\nPosition du vrai investisseur (indice 0): RANG {positive_rank.item()}")

    print(f"❌ Très mauvaise prédiction!")
    print(f"   MRR contribution: 1/{positive_rank.item()} = {1.0/positive_rank.item():.4f}")
    print(f"   Recall@1: False")
    print(f"   Recall@10: False")
    print(f"   Recall@50: {positive_rank.item() <= 50}")

    # ============================================================
    # BATCH EXAMPLE: Calculer MRR et Recall sur plusieurs exemples
    # ============================================================
    print("\n\n" + "=" * 70)
    print("BATCH EXAMPLE: 3 startups ensemble")
    print("=" * 70)

    # 3 startups avec leurs scores
    # Startup 1: vrai investisseur classé 1er (parfait)
    # Startup 2: vrai investisseur classé 5ème (moyen)
    # Startup 3: vrai investisseur classé 50ème (mauvais)

    pos_scores_batch = torch.tensor([[0.78], [0.35], [0.05]])  # 3 vrais investisseurs
    neg_scores_batch = torch.cat([
        torch.tensor([[0.65, 0.23, 0.19]]),  # Startup 1: 3 négatifs
        torch.tensor([[0.78, 0.65, 0.52]]),  # Startup 2: 3 négatifs
        torch.linspace(0.95, 0.06, 3).unsqueeze(0)  # Startup 3: 3 négatifs
    ], dim=0)

    all_scores_batch = torch.cat([pos_scores_batch, neg_scores_batch], dim=1)
    rankings_batch = torch.argsort(all_scores_batch, dim=1, descending=True)
    positive_ranks_batch = (rankings_batch == 0).nonzero(as_tuple=True)[1] + 1

    print(f"\nRangs des vrais investisseurs: {positive_ranks_batch.numpy()}")
    print(f"  Startup 1: rang {positive_ranks_batch[0].item()}")
    print(f"  Startup 2: rang {positive_ranks_batch[1].item()}")
    print(f"  Startup 3: rang {positive_ranks_batch[2].item()}")

    # MRR
    mrr = (1.0 / positive_ranks_batch.float()).mean().item()
    print(f"\nMRR = moyenne(1/rang) = moyenne({[(1.0/r.item()) for r in positive_ranks_batch]})")
    print(f"    = {mrr:.3f}")

    # Recall@K
    for k in [1, 3, 10]:
        recall = (positive_ranks_batch <= k).float().mean().item()
        count = (positive_ranks_batch <= k).sum().item()
        print(f"Recall@{k} = {count}/3 = {recall:.2%}")

    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    print("✓ Si rang = 1 → Top proba = Vrai prédiction")
    print("✓ Indice 0 dans all_scores = TOUJOURS le vrai investisseur")
    print("✓ argsort(descending=True) classe du meilleur au pire")
    print("✓ (rankings == 0).nonzero() trouve la position du vrai")
    print("=" * 70)


if __name__ == "__main__":
    test_ranking_mechanism()
