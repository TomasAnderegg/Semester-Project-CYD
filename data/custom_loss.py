import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from pathlib import Path

class SimpleBusinessWeightedLoss(nn.Module):
    """
    Version SIMPLE et efficace : BCE weighted par les degrés des entreprises
    Sans focal loss, sans complications - juste des poids métier intelligents
    """
    
    def __init__(self, company_degrees, company_to_id, 
                 pos_weight=5.0,  # Poids pour les positifs (réduit de 169 à 5)
                 diversity_boost=1.3,  # Boost léger pour entreprises peu connues
                 use_diversity=False,
                 max_company_id=None):  # AJOUT: Pour gérer les IDs dispersés
        super().__init__()
        
        self.pos_weight = pos_weight
        self.diversity_boost = diversity_boost
        self.use_diversity = use_diversity
        
        # FIX: Déterminer la taille du tensor de poids
        if max_company_id is not None:
            weight_size = max_company_id + 1
        else:
            weight_size = max(company_to_id.values()) + 1
        
        # Calculer les poids par entreprise (si activé)
        if use_diversity:
            self.company_weights = self._compute_weights(company_degrees, company_to_id, weight_size)
        else:
            self.company_weights = torch.ones(weight_size)
        
        print(f"\n{'='*70}")
        print("SIMPLE BUSINESS WEIGHTED LOSS")
        print(f"{'='*70}")
        print(f"  Positive weight: {pos_weight}")
        print(f"  Diversity boost: {diversity_boost if use_diversity else 'disabled'}")
        print(f"  Weight tensor size: {weight_size} (to handle IDs 0-{weight_size-1})")
        print(f"  Company weights range: [{self.company_weights.min():.2f}, {self.company_weights.max():.2f}]")
    
    def _compute_weights(self, company_degrees, company_to_id, weight_size):
        """Poids simples basés sur la popularité"""
        # Créer un tensor de la bonne taille pour gérer tous les IDs
        weights = torch.ones(weight_size)
        
        for company, cid in company_to_id.items():
            if cid >= weight_size:
                continue  # Skip les IDs hors limites (ne devrait pas arriver)
            
            degree = company_degrees.get(company, 1)
            
            # Stratégie simple : boost les entreprises peu connues
            if degree <= 5:  # Peu connu
                weights[cid] = self.diversity_boost
            elif degree > 50:  # Trop connu
                weights[cid] = 1.0 / self.diversity_boost  # Légère pénalité
            else:  # Normal
                weights[cid] = 1.0
        
        return weights
    
    def forward(self, pred_logits, targets, company_ids=None):
        """
        Args:
            pred_logits: logits du modèle (avant sigmoid)
            targets: labels (0/1)
            company_ids: IDs des entreprises
        """
        # 1. BCE de base avec pos_weight
        pos_weight_tensor = torch.tensor(self.pos_weight, device=targets.device, dtype=targets.dtype)
        
        # BCEWithLogitsLoss avec pos_weight
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, 
            targets, 
            pos_weight=pos_weight_tensor if (targets == 1).any() else None,
            reduction='none'
        )
        
        # 2. Appliquer les poids métier si activé
        if self.use_diversity and company_ids is not None:
            # FIX: Gérer les IDs qui dépassent la taille du tensor
            max_weight_id = self.company_weights.size(0)
            
            # Clamp les IDs pour éviter les out of bounds
            safe_ids = torch.clamp(company_ids, 0, max_weight_id - 1)
            
            # Log un warning si on a des IDs hors limites (debug)
            if (company_ids >= max_weight_id).any():
                n_oob = (company_ids >= max_weight_id).sum().item()
                print(f"⚠️  Warning: {n_oob} company IDs are out of bounds (max={max_weight_id})")
            
            company_weight = self.company_weights[safe_ids].to(pred_logits.device)
            bce_loss = bce_loss * company_weight
        
        return bce_loss.mean()


def load_company_degrees_and_map(data_name="crunchbase"):
    """Charge les degrés et le mapping des entreprises"""
    
    # 1. Charger les degrés
    degree_path = f"data/{data_name}_filtered_company_degrees.pkl"
    if not Path(degree_path).exists():
        print(f"⚠️  Fichier de degrés introuvable: {degree_path}")
        return None, None
    
    with open(degree_path, "rb") as f:
        company_degrees = pickle.load(f)
    
    # 2. Charger le mapping company -> id
    map_path = f"data/mappings/{data_name}_filtered_company_id_map.pickle"
    if not Path(map_path).exists():
        print(f"⚠️  Fichier de mapping introuvable: {map_path}")
        return company_degrees, None
    
    with open(map_path, "rb") as f:
        company_to_id = pickle.load(f)
    
    # 3. IMPORTANT: Charger aussi le mapping inverse (id -> company) pour détecter tous les IDs
    inverse_map_path = f"data/mappings/{data_name}_filtered_id_company_map.pickle"
    max_possible_id = max(company_to_id.values()) + 1
    
    if Path(inverse_map_path).exists():
        with open(inverse_map_path, "rb") as f:
            id_to_company = pickle.load(f)
            max_possible_id = max(id_to_company.keys()) + 1
            print(f"✓ Détecté {max_possible_id} IDs possibles (via inverse map)")
    
    print(f"✓ Chargé {len(company_degrees)} degrés et {len(company_to_id)} mappings")
    print(f"✓ Range des IDs: 0 à {max_possible_id - 1}")
    
    return company_degrees, company_to_id


def create_simple_business_loss(data_name="crunchbase", phase=1, max_company_id=None):
    """
    Crée une loss simple et efficace
    
    Args:
        phase: 1=baseline (pos_weight=5), 2=with diversity (pos_weight=5 + diversity)
        max_company_id: Le max ID possible dans les données (pour éviter index errors)
    """
    
    company_degrees, company_to_id = load_company_degrees_and_map(data_name)
    
    if company_degrees is None or company_to_id is None:
        print("⚠️  Utilisation de BCEWithLogitsLoss standard")
        return nn.BCEWithLogitsLoss()
    
    # Filtrer les degrés
    filtered_degrees = {
        name: company_degrees[name] 
        for name in company_to_id.keys() 
        if name in company_degrees
    }
    
    if phase == 1:
        print("🎯 PHASE 1: Baseline (pos_weight seulement)")
        return SimpleBusinessWeightedLoss(
            company_degrees=filtered_degrees,
            company_to_id=company_to_id,
            pos_weight=5.0,  # Léger boost pour positifs
            diversity_boost=1.3,
            use_diversity=False,  # Désactivé en phase 1
            max_company_id=max_company_id
        )
    
    elif phase == 2:
        print("🎯 PHASE 2: Avec diversité")
        return SimpleBusinessWeightedLoss(
            company_degrees=filtered_degrees,
            company_to_id=company_to_id,
            pos_weight=5.0,
            diversity_boost=1.3,  # Boost léger pour entreprises peu connues
            use_diversity=True,  # Activé en phase 2
            max_company_id=max_company_id
        )
    
    else:  # phase 3
        print("🎯 PHASE 3: Diversité agressive")
        return SimpleBusinessWeightedLoss(
            company_degrees=filtered_degrees,
            company_to_id=company_to_id,
            pos_weight=5.0,
            diversity_boost=1.5,  # Boost plus fort
            use_diversity=True,
            max_company_id=max_company_id
        )