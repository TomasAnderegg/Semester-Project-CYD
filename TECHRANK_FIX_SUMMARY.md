# Fix TechRank: Élimination des Self-Loops

## Problème Identifié

Le graphe bipartite contenait des **self-loops** et des **assignments bipartite incorrects** causant des scores TechRank à 0.0.

### Cause Racine

Des entités comme "Legend Capital" apparaissaient dans **BOTH** `id_to_company` ET `id_to_investor`:

```python
# Avant le fix:
company_name = node_name.get(123, "company_123")  # → "Legend Capital"
investor_name = node_name.get(456, "investor_456")  # → "Legend Capital"

# Création d'un self-loop!
pred_graph.add_edge(company_name, investor_name)  # Legend Capital → Legend Capital
```

Cela créait:
1. **Self-loops**: Même nom utilisé comme source ET destination
2. **Bipartite incorrect**: Le dernier `add_node()` écrasait l'attribut bipartite
3. **Structure invalide**: TechRank nécessite un graphe bipartite strict

## Solution Implémentée

### 1. Préfixage des Node IDs (TGN_eval.py:940-947)

Chaque nom est maintenant préfixé avec son rôle:

```python
# Récupérer les noms de base
company_base_name = node_name.get(s, f"company_{s}")
investor_base_name = node_name.get(d, f"investor_{d}")

# ⚠️ CRITIQUE: Préfixer pour éviter collisions
company_name = f"COMPANY_{company_base_name}"
investor_name = f"INVESTOR_{investor_base_name}"
```

**Résultat:**
- `COMPANY_Legend Capital` ≠ `INVESTOR_Legend Capital`
- Plus de self-loops!
- Chaque node a un ID unique

### 2. Stockage du Nom de Base (TGN_eval.py:950-966)

Les dictionnaires stockent maintenant les deux versions:

```python
dict_companies[company_name] = {
    'id': company_id,
    'name': company_name,          # Avec préfixe (pour graphe)
    'base_name': company_base_name,  # Sans préfixe (pour affichage)
    ...
}
```

### 3. Affichage Sans Préfixe (TGN_eval.py:1228-1230, 1246-1248)

L'affichage final enlève les préfixes pour montrer les noms originaux:

```python
# Pour les investors
display_name = row['final_configuration'].replace("INVESTOR_", "")
logger.info(f"   #{idx:2d} {display_name:40s} → Score: {row['techrank']:.6f}")

# Pour les companies
display_name = row['final_configuration'].replace("COMPANY_", "")
logger.info(f"   #{idx:2d} {display_name:40s} → Score: {row['techrank']:.6f}")
```

## Fichiers Modifiés

### 1. [TGN_eval.py](TGN_eval.py)

- **Lignes 940-947**: Préfixage des node IDs
- **Lignes 950-966**: Stockage base_name dans dictionnaires
- **Lignes 1228-1230**: Affichage investors sans préfixe
- **Lignes 1246-1248**: Affichage companies sans préfixe

### 2. [verify_fixed_graph.py](verify_fixed_graph.py) (nouveau)

Script de vérification qui:
- Vérifie l'absence de self-loops
- Confirme la structure bipartite correcte
- Compte les préfixes COMPANY_ et INVESTOR_
- Affiche des exemples d'arêtes

## Comment Tester

### Étape 1: Lancer TGN_eval.py

```bash
python TGN_eval.py --use_memory --prefix tgn-crunchbase --n_runs 1 --prediction_threshold 0.0
```

### Étape 2: Vérifier le graphe généré

```bash
python verify_fixed_graph.py
```

**Résultat attendu:**
```
✅ SUCCÈS: Le graphe est valide!
   - Aucun self-loop
   - Structure bipartite respectée
   - Les préfixes sont correctement appliqués
```

### Étape 3: Vérifier les scores TechRank

Dans la sortie de TGN_eval.py, chercher:

```
📊 Résultats Investors:
   Total: XXX
   Scores > 0: YYY  # ← Devrait être > 0 maintenant!
   Score max: Z.ZZZZZZ

📊 Top 10 Investors (par TechRank):
   # 1 Legend Capital                           → Score: 0.XXXXXX
   # 2 Sequoia Capital                          → Score: 0.XXXXXX
   ...
```

## Vérifications Importantes

### 1. Plus de self-loops

```python
# AVANT (MAUVAIS):
Legend Capital (bipartite=1) → Legend Capital (bipartite=1)

# APRÈS (BON):
COMPANY_Legend Capital (bipartite=0) → INVESTOR_Accel (bipartite=1)
```

### 2. Bipartite assignments corrects

```
Companies dans bipartite=0: 422/422 ✅
Companies dans bipartite=1: 0/422 ✅

Investors dans bipartite=0: 0/224 ✅
Investors dans bipartite=1: 224/224 ✅
```

### 3. Scores TechRank non-nuls

```
Investors avec score > 0: XXX/224 (devrait être > 1)
Companies avec score > 0: YYY/422 (devrait être > 1)
```

## Convention Bipartite Maintenue

La convention utilisateur est **préservée**:

- `bipartite=0` → **Companies** (sources dans TGN)
- `bipartite=1` → **Investors** (destinations dans TGN)
- Edges: `Company → Investor`

## Prochaines Étapes

1. **Tester** avec `python TGN_eval.py --use_memory --prefix tgn-crunchbase --n_runs 1`
2. **Vérifier** avec `python verify_fixed_graph.py`
3. **Confirmer** que les scores TechRank sont non-nuls
4. Si problème persiste, vérifier les logs pour identifier la nouvelle cause

## Notes Techniques

### Pourquoi les préfixes dans le graphe?

NetworkX utilise les node IDs comme clés dans un dictionnaire. Si deux nodes ont le même ID (même nom), NetworkX les considère comme **le même node**. Les préfixes garantissent l'unicité.

### Pourquoi enlever les préfixes à l'affichage?

Pour l'utilisateur, "Legend Capital" est plus lisible que "INVESTOR_Legend Capital". Le préfixe est un détail d'implémentation interne.

### Compatibilité avec TechRank

TechRank.py utilise les clés des dictionnaires comme node names. Tant que:
1. Les clés des dictionnaires correspondent aux node IDs du graphe
2. La structure bipartite est respectée (bipartite=0 et bipartite=1)

TechRank fonctionnera correctement, peu importe le format des noms.
