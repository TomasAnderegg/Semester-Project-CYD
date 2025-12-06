import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path

class InverseDegreeWeightedBCELoss(nn.Module):
    """
    BCELoss pondérée par 1/degré pour favoriser les entreprises à faible degré.
    
    Logique:
    - Entreprise avec degré 1 → poids élevé
    - Entreprise avec degré 100 → poids faible
    - Investisseur → poids = 1.0 (pas de pondération)
    """
    
    def __init__(self, company_degrees, item_map, alpha=1.0, normalize=True):
        """
        Args:
            company_degrees: dict {company_name: degree}
            item_map: dict {company_name: company_id}
            alpha: Exposant pour contrôler l'intensité (alpha=1 → linéaire, alpha>1 → plus agressif)
            normalize: Si True, normalise les poids pour avoir une moyenne de 1.0
        """
        super().__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCELoss(reduction='none')  # reduction='none' pour pondérer manuellement
        
        # Créer un tenseur de poids indexé par company_id
        max_id = max(item_map.values()) + 1
        self.weights = torch.ones(max_id, dtype=torch.float32)
        
        inverse_degrees = []
        for company_name, company_id in item_map.items():
            degree = company_degrees.get(company_name, 1)
            inv_degree = (1.0 / degree) ** alpha
            self.weights[company_id] = inv_degree
            inverse_degrees.append(inv_degree)
        
        # Normalisation pour que la moyenne des poids = 1.0
        if normalize:
            mean_weight = np.mean(inverse_degrees)
            self.weights = self.weights / mean_weight
        
        print(f"\n{'='*70}")
        print("WEIGHTED LOSS CONFIGURATION")
        print(f"{'='*70}")
        print(f"  Alpha (exponent): {alpha}")
        print(f"  Normalize weights: {normalize}")
        print(f"  Weight statistics:")
        print(f"    - Min: {self.weights.min().item():.4f}")
        print(f"    - Max: {self.weights.max().item():.4f}")
        print(f"    - Mean: {self.weights.mean().item():.4f}")
        print(f"    - Median: {self.weights.median().item():.4f}")
    
    def forward(self, predictions, targets, node_ids):
        """
        Args:
            predictions: Tensor de prédictions (batch_size,)
            targets: Tensor de labels (batch_size,)
            node_ids: Tensor des IDs des companies (batch_size,) - correspond à 'u' dans TGN
        
        Returns:
            Weighted loss (scalar)
        """
        # Calculer la loss de base (par exemple)
        base_loss = self.bce_loss(predictions, targets)
        
        # Récupérer les poids pour chaque nœud du batch
        weights = self.weights[node_ids].to(predictions.device)
        
        # Appliquer les poids
        weighted_loss = base_loss * weights
        
        # Retourner la moyenne
        return weighted_loss.mean()


def prepare_degree_weights(B, item_map, output_prefix="crunchbase_filtered"):
    """
    Prépare et sauvegarde les degrés des entreprises pour la weighted loss.
    
    Args:
        B: Graphe bipartite
        item_map: Mapping {company_name: company_id}
        output_prefix: Préfixe pour les fichiers de sortie
    
    Returns:
        company_degrees: dict {company_name: degree}
    """
    print(f"\n{'='*70}")
    print("PRÉPARATION DES DEGRÉS POUR WEIGHTED LOSS")
    print(f"{'='*70}")
    
    company_degrees = {}
    
    for company_name, company_id in item_map.items():
        degree = B.degree(company_name)
        company_degrees[company_name] = degree
    
    # Statistiques
    degrees = list(company_degrees.values())
    print(f"\n  Statistiques des degrés:")
    print(f"    - Min: {min(degrees)}")
    print(f"    - Max: {max(degrees)}")
    print(f"    - Moyenne: {np.mean(degrees):.2f}")
    print(f"    - Médiane: {np.median(degrees):.0f}")
    
    # Distribution
    print(f"\n  Distribution:")
    print(f"    - Degré <= 5: {sum(1 for d in degrees if d <= 5)} entreprises ({100*sum(1 for d in degrees if d <= 5)/len(degrees):.1f}%)")
    print(f"    - Degré <= 10: {sum(1 for d in degrees if d <= 10)} entreprises ({100*sum(1 for d in degrees if d <= 10)/len(degrees):.1f}%)")
    print(f"    - Degré > 50: {sum(1 for d in degrees if d > 50)} entreprises ({100*sum(1 for d in degrees if d > 50)/len(degrees):.1f}%)")
    
    # Sauvegarder
    Path("data").mkdir(exist_ok=True)
    with open(f"data/{output_prefix}_company_degrees.pkl", "wb") as f:
        pickle.dump(company_degrees, f)
    
    print(f"\n✓ Degrés sauvegardés: data/{output_prefix}_company_degrees.pkl")
    
    return company_degrees


# =============================================================================
# MODIFICATIONS POUR train.py
# =============================================================================

def load_weighted_loss(data_name, item_map, alpha=1.0, normalize=True):
    """
    Charge les degrés et crée la weighted loss.
    À appeler dans train.py après avoir chargé les données.
    
    Args:
        data_name: Nom du dataset (ex: 'crunchbase')
        item_map: Mapping chargé depuis les pickles
        alpha: Exposant de pondération
        normalize: Normaliser les poids
    
    Returns:
        criterion: InverseDegreeWeightedBCELoss
    """
    degree_path = f"data/{data_name}_filtered_company_degrees.pkl"
    
    if not Path(degree_path).exists():
        print(f"⚠️  Fichier de degrés introuvable: {degree_path}")
        print("   Utilisation de BCELoss standard à la place")
        return nn.BCELoss()
    
    with open(degree_path, "rb") as f:
        company_degrees = pickle.load(f)
    
    # Inverser item_map pour avoir {company_id: company_name}
    id_to_company = {v: k for k, v in item_map.items()}
    
    # Créer un dict {company_name: degree} filtré
    filtered_degrees = {name: company_degrees[name] 
                        for name in item_map.keys() 
                        if name in company_degrees}
    
    criterion = InverseDegreeWeightedBCELoss(
        company_degrees=filtered_degrees,
        item_map=item_map,
        alpha=alpha,
        normalize=normalize
    )
    
    return criterion


# =============================================================================
# EXEMPLE D'INTÉGRATION DANS LA BOUCLE D'ENTRAÎNEMENT
# =============================================================================

def train_step_with_weighted_loss(tgn, optimizer, criterion, sources_batch, 
                                   destinations_batch, negatives_batch, 
                                   timestamps_batch, edge_idxs_batch, 
                                   NUM_NEIGHBORS, device):
    """
    Exemple de fonction d'entraînement modifiée pour utiliser la weighted loss.
    
    IMPORTANT: sources_batch correspond aux companies (u), destinations_batch aux investors (i)
    """
    size = len(sources_batch)
    
    pos_label = torch.ones(size, dtype=torch.float, device=device)
    neg_label = torch.zeros(size, dtype=torch.float, device=device)
    
    tgn = tgn.train()
    pos_prob, neg_prob = tgn.compute_edge_probabilities(
        sources_batch, destinations_batch, negatives_batch,
        timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS
    )
    
    # Convertir sources_batch en tensor pour indexation
    sources_tensor = torch.from_numpy(sources_batch).long().to(device)
    
    # Si criterion est weighted, passer les node_ids
    if isinstance(criterion, InverseDegreeWeightedBCELoss):
        # Loss pour les liens positifs
        pos_loss = criterion(pos_prob.squeeze(), pos_label, sources_tensor)
        
        # Loss pour les liens négatifs (même poids car même source)
        neg_loss = criterion(neg_prob.squeeze(), neg_label, sources_tensor)
        
        loss = pos_loss + neg_loss
    else:
        # BCELoss standard
        loss = criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
    
    return loss


# =============================================================================
# COMPARAISON: Node Feature vs Weighted Loss
# =============================================================================

def comparison_table():
    """
    Affiche un tableau comparatif des deux approches.
    """
    print(f"\n{'='*80}")
    print("COMPARAISON: NODE FEATURE (1/degré) vs WEIGHTED LOSS")
    print(f"{'='*80}\n")
    
    comparison = """
┌─────────────────────────┬──────────────────────────┬──────────────────────────┐
│ Critère                 │ Node Feature (1/degré)   │ Weighted Loss            │
├─────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Explicité               │ Indirecte - le modèle    │ Directe - dit au modèle  │
│                         │ doit apprendre           │ quoi privilégier         │
├─────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Garantie d'effet        │ ❌ Non - dépend des      │ ✅ Oui - force le modèle │
│                         │ patterns dans les données│ à accorder plus de poids │
├─────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Flexibilité             │ ✅ Le modèle peut        │ ⚠️  Moins flexible - vous │
│                         │ apprendre des patterns   │ imposez le biais         │
│                         │ complexes                │                          │
├─────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Surapprentissage        │ ⚠️  Risque si le degré   │ ⚠️  Risque de trop       │
│                         │ n'est pas vraiment       │ pénaliser les entreprises│
│                         │ prédictif                │ populaires               │
├─────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Complexité              │ Simple à implémenter     │ Nécessite modification   │
│ d'implémentation        │                          │ de la loss function      │
├─────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Interprétabilité        │ ✅ Facile - c'est une    │ ✅ Très clair - pondère  │
│                         │ feature comme une autre  │ directement l'erreur     │
├─────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Cas d'usage idéal       │ Quand le degré est une   │ Quand vous SAVEZ que les │
│                         │ info parmi d'autres      │ faibles degrés sont plus │
│                         │                          │ importants               │
└─────────────────────────┴──────────────────────────┴──────────────────────────┘

RECOMMANDATION:
• Si vous n'êtes PAS SÛR que faible degré = meilleure opportunité
  → Node Feature (laisse le modèle apprendre)

• Si vous ÊTES SÛR que vous voulez favoriser les faibles degrés
  → Weighted Loss (impose votre connaissance métier)

• IDÉAL: Tester les DEUX et comparer avec W&B!
    """
    print(comparison)


if __name__ == "__main__":
    comparison_table()