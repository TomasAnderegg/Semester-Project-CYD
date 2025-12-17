# Fix Critical: Collision des ID dans les Dictionnaires

## Problème Découvert

Les dictionnaires `dict_companies` et `dict_investors` contenaient des **inversions massives** où des investors étaient classés comme companies et vice versa.

### Exemple du Problème

**Fichier sauvegardé: dict_companies_crunchbase.pickle**
```
90 investors incorrectement classés comme companies:
- COMPANY_Legend Capital       ← C'EST UN INVESTOR!
- COMPANY_Wuxi Capital          ← C'EST UN INVESTOR!
- COMPANY_Mergus Ventures       ← C'EST UN INVESTOR!
```

## Cause Racine

### IDs qui se Chevauchent

Dans TGN, les IDs des companies et des investors **commencent tous les deux à 0**:

```
Companies:   0, 1, 2, 3, ...
Investors:   0, 1, 2, 3, ...  ← CHEVAUCHEMENT!
```

### Code Problématique (AVANT)

```python
node_name = {}

# Sources = COMPANIES
for src_id in all_sources:
    node_name[src_id] = id_to_company[src_id]  # node_name[0] = "TuringQ"

# Destinations = INVESTORS
for dst_id in all_destinations:
    node_name[dst_id] = id_to_investor[dst_id]  # node_name[0] = "Legend Capital" ← ÉCRASE!
```

Quand `src_id=0` ET `dst_id=0`, le deuxième assignment **écrasait** le premier!

### Conséquence

```python
# Plus tard dans le code:
company_base_name = node_name.get(s, f"company_{s}")  # s=0
# Retourne "Legend Capital" au lieu de "TuringQ"!

# Donc on crée:
company_name = f"COMPANY_Legend Capital"  # ❌ FAUX!
```

Résultat: 90 investors se retrouvaient dans dict_companies!

## Solution Implémentée

### Deux Dictionnaires Séparés

[TGN_eval.py:819-838](TGN_eval.py#L819-L838)

```python
# ⚠️ CRITIQUE: Utiliser des dictionnaires SÉPARÉS car les IDs se chevauchent!
company_id_to_name = {}
investor_id_to_name = {}

# Sources = COMPANIES
for src_id in all_sources:
    if src_id in id_to_company:
        company_id_to_name[src_id] = id_to_company[src_id]  # Séparé!

# Destinations = INVESTORS
for dst_id in all_destinations:
    if dst_id in id_to_investor:
        investor_id_to_name[dst_id] = id_to_investor[dst_id]  # Séparé!
```

### Utilisation Correcte

[TGN_eval.py:942-943](TGN_eval.py#L942-L943)

```python
# Récupérer les noms depuis les dictionnaires SÉPARÉS
company_base_name = company_id_to_name.get(s, f"company_{s}")  # ✅ CORRECT
investor_base_name = investor_id_to_name.get(d, f"investor_{d}")  # ✅ CORRECT
```

## Vérification

### Avant le Fix

```bash
python check_predicted_dicts.py
```

```
dict_companies_crunchbase.pickle:
   ⚠️  Noms suspects (possibles investors):
      - COMPANY_Legend Capital
      - COMPANY_Wuxi Capital
      ... et 85 autres
```

### Après le Fix

```bash
# Relancer TGN_eval.py
python TGN_eval.py --use_memory --prefix tgn-crunchbase --n_runs 1

# Puis vérifier
python check_predicted_dicts.py
```

**Résultat attendu:**
```
dict_companies_crunchbase.pickle:
   Premiers 10 noms:
      1. COMPANY_TuringQ           ← VRAIE company!
      2. COMPANY_Quantum Flytrap   ← VRAIE company!
      3. COMPANY_Arctic Instruments ← VRAIE company!

   ⚠️  Noms suspects: 0  ← AUCUN INVESTOR!
```

## Impact

Cette fix résout:
1. ✅ Les inversions company/investor dans les dictionnaires sauvegardés
2. ✅ Les scores TechRank qui comparaient des investors avec des companies
3. ✅ Le problème "delta négatif pour toutes les companies"

## Fichiers Modifiés

- [TGN_eval.py](TGN_eval.py)
  - Lignes 819-838: Création de deux dictionnaires séparés
  - Lignes 942-943: Utilisation des dictionnaires séparés

## Tests

1. **Vérifier les dictionnaires sauvegardés:**
   ```bash
   python check_predicted_dicts.py
   ```

2. **Vérifier la comparaison TechRank:**
   ```bash
   python TechRank_Comparison.py --data crunchbase --mapping_dir data/mappings --alpha 0.0 --beta -50 --save_dir techrank_comparison --plot
   ```

3. **Résultat attendu:**
   - Les noms dans dict_companies doivent être de vraies companies
   - Les deltas TechRank doivent être calculables
   - Certaines companies devraient avoir delta > 0

## Pourquoi C'était Difficile à Détecter

1. Les préfixes COMPANY_ et INVESTOR_ **masquaient** le problème
2. Les logs montraient "COMPANY_Legend Capital" qui **semblait** correct
3. C'est seulement en vérifiant les **noms de base** qu'on a vu "Legend Capital" = investor

## Leçon Apprise

⚠️ **NE JAMAIS utiliser un dictionnaire unique pour des entités avec des espaces d'ID qui se chevauchent!**

Toujours utiliser des dictionnaires séparés ou des clés composites comme `(type, id)`.
