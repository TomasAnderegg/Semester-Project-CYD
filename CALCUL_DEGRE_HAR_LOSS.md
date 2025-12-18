# Calcul du Degré pour HAR Loss

## Question : Comment est calculé le degré dans HAR Loss ?

Le degré d'un nœud représente le **nombre total d'interactions (edges)** auxquelles ce nœud participe dans le **training set uniquement**.

---

## Implémentation : `build_degree_dict()`

### Code Source ([har_loss.py:202-226](har_loss.py#L202-L226))

```python
def build_degree_dict(data):
    """
    Build a dictionary mapping node IDs to their degrees.

    Args:
        data (Data): Graph data with sources and destinations

    Returns:
        dict: {node_id: degree}
    """
    from collections import Counter

    # Count occurrences of each node
    degree_dict = Counter()

    # Count source degrees
    for src in data.sources:
        degree_dict[int(src)] += 1

    # Count destination degrees
    for dst in data.destinations:
        degree_dict[int(dst)] += 1

    return dict(degree_dict)
```

### Explications Ligne par Ligne

```python
degree_dict = Counter()  # Initialise un compteur vide
```

**1. Compter les sources :**
```python
for src in data.sources:
    degree_dict[int(src)] += 1
```
- Parcourt toutes les interactions du training set
- Pour chaque edge `(src, dst, timestamp)`, incrémente le degré de `src`

**2. Compter les destinations :**
```python
for dst in data.destinations:
    degree_dict[int(dst)] += 1
```
- Parcourt toutes les destinations du training set
- Pour chaque edge, incrémente aussi le degré de `dst`

**Résultat :** Chaque nœud a un degré = nombre total d'apparitions (comme source OU destination)

---

## Graphe Non-Dirigé (Undirected)

**Important :** Le degré est calculé en **mode non-dirigé** :
- Chaque edge contribue au degré des **deux nœuds** (source ET destination)
- Si `Company A` → `Investor B`, alors :
  - `degree(Company A) += 1`
  - `degree(Investor B) += 1`

### Pourquoi Non-Dirigé ?

Dans un graphe biparti Company-Investor :
- Une startup peut recevoir plusieurs investissements (haut degré = populaire)
- Un VC peut investir dans plusieurs startups (haut degré = actif)
- Les deux directions sont pertinentes pour mesurer la "popularité"

---

## Exemple Concret : CrunchBase

### Dataset Training

Supposons ces 10 interactions dans le training set :

| ID | Source (Startup) | Destination (Investor) | Timestamp |
|----|------------------|------------------------|-----------|
| 1  | DeepMind (42)    | Google Ventures (100)  | 2010-01   |
| 2  | DeepMind (42)    | Founders Fund (101)    | 2010-06   |
| 3  | DeepMind (42)    | Horizon Ventures (102) | 2011-03   |
| 4  | OpenAI (43)      | Y Combinator (103)     | 2015-12   |
| 5  | OpenAI (43)      | Microsoft (104)        | 2019-07   |
| 6  | Stripe (44)      | Y Combinator (103)     | 2010-09   |
| 7  | Stripe (44)      | Sequoia (105)          | 2011-02   |
| 8  | Airbnb (45)      | Sequoia (105)          | 2011-04   |
| 9  | Airbnb (45)      | Y Combinator (103)     | 2009-01   |
| 10 | StartupX (46)    | Angel Investor (106)   | 2023-05   |

### Calcul du Degré

#### Startups (Companies)

```python
# DeepMind (42): apparaît comme source dans edges 1, 2, 3
degree(42) = 3

# OpenAI (43): apparaît comme source dans edges 4, 5
degree(43) = 2

# Stripe (44): apparaît comme source dans edges 6, 7
degree(44) = 2

# Airbnb (45): apparaît comme source dans edges 8, 9
degree(45) = 2

# StartupX (46): apparaît comme source dans edge 10
degree(46) = 1  ← Startup émergente (low-degree)
```

#### Investisseurs (Investors)

```python
# Google Ventures (100): apparaît comme destination dans edge 1
degree(100) = 1

# Founders Fund (101): apparaît comme destination dans edge 2
degree(101) = 1

# Horizon Ventures (102): apparaît comme destination dans edge 3
degree(102) = 1

# Y Combinator (103): apparaît comme destination dans edges 4, 6, 9
degree(103) = 3  ← VC populaire

# Microsoft (104): apparaît comme destination dans edge 5
degree(104) = 1

# Sequoia (105): apparaît comme destination dans edges 7, 8
degree(105) = 2

# Angel Investor (106): apparaît comme destination dans edge 10
degree(106) = 1
```

### Dictionnaire Final

```python
degree_dict = {
    42: 3,   # DeepMind
    43: 2,   # OpenAI
    44: 2,   # Stripe
    45: 2,   # Airbnb
    46: 1,   # StartupX ← LOW-DEGREE
    100: 1,  # Google Ventures
    101: 1,  # Founders Fund
    102: 1,  # Horizon Ventures
    103: 3,  # Y Combinator ← HIGH-DEGREE
    104: 1,  # Microsoft
    105: 2,  # Sequoia
    106: 1,  # Angel Investor
}
```

---

## Conversion en Tensor (Optimisation)

### Code ([train_self_supervised.py:178-189](train_self_supervised.py#L178-L189))

```python
# Build degree dictionary
degree_dict = build_degree_dict(train_data)

# Convert to tensor for O(1) lookup during training
max_node_id = max(degree_dict.keys())  # 106 dans notre exemple
degree_tensor = torch.zeros(max_node_id + 1, dtype=torch.float32)

for node_id, degree in degree_dict.items():
    degree_tensor[node_id] = degree
```

**Résultat :**
```python
degree_tensor = [
    0,   # Node 0 (n'existe pas)
    0,   # Node 1 (n'existe pas)
    ...
    0,   # Node 41 (n'existe pas)
    3.0, # Node 42 (DeepMind)
    2.0, # Node 43 (OpenAI)
    2.0, # Node 44 (Stripe)
    2.0, # Node 45 (Airbnb)
    1.0, # Node 46 (StartupX) ← LOW-DEGREE
    ...
    1.0, # Node 100 (Google Ventures)
    1.0, # Node 101 (Founders Fund)
    1.0, # Node 102 (Horizon Ventures)
    3.0, # Node 103 (Y Combinator) ← HIGH-DEGREE
    1.0, # Node 104 (Microsoft)
    2.0, # Node 105 (Sequoia)
    1.0, # Node 106 (Angel Investor)
]
```

**Avantage :** Lookup en O(1) durant l'entraînement via `degree_tensor[node_id]`

---

## Utilisation dans HAR Loss

### Lookup Durant Training

Pendant l'entraînement, pour chaque batch :

```python
# Batch de 32 paires (src, dst)
sources_batch = [46, 43, 42, ...]  # IDs des startups
destinations_batch = [106, 105, 103, ...]  # IDs des investisseurs

# Lookup rapide des degrés
src_degrees = degree_tensor[sources_batch]
# → [1.0, 2.0, 3.0, ...]

dst_degrees = degree_tensor[destinations_batch]
# → [1.0, 2.0, 3.0, ...]
```

### Calcul des Poids

```python
# HAR Loss calcule les poids (alpha = 0.5)
w_src = src_degrees ** (-0.5)
# → [1.0^(-0.5), 2.0^(-0.5), 3.0^(-0.5), ...]
# → [1.000, 0.707, 0.577, ...]

w_dst = dst_degrees ** (-0.5)
# → [1.000, 0.707, 0.577, ...]

# Poids combiné pour chaque paire
w_pair = w_src * w_dst
```

### Exemple avec StartupX

```python
# Paire: StartupX (46) → Angel Investor (106)
src_degree = 1.0
dst_degree = 1.0

# Poids HAR (alpha = 0.5)
w_pair = (1.0)^(-0.5) × (1.0)^(-0.5) = 1.0 × 1.0 = 1.0

# Perte HAR
loss_har = 1.0 × contrastive_loss  # Poids MAXIMAL
```

### Exemple avec DeepMind

```python
# Paire: DeepMind (42) → Y Combinator (103)
src_degree = 3.0
dst_degree = 3.0

# Poids HAR (alpha = 0.5)
w_pair = (3.0)^(-0.5) × (3.0)^(-0.5) = 0.577 × 0.577 = 0.333

# Perte HAR
loss_har = 0.333 × contrastive_loss  # Poids RÉDUIT (3× moins que StartupX)
```

---

## Propriétés Importantes

### 1. Degré = Popularité dans Training Set

```
High-degree node = Nœud qui apparaît souvent dans training
Low-degree node = Nœud rare dans training

Exemples:
  - Y Combinator (degree=3): A investi dans 3 startups du training
  - StartupX (degree=1): N'a qu'un seul investissement dans training
```

### 2. Graphe Biparti : Degrés Asymétriques

Dans CrunchBase, le graphe est **biparti** :
- Companies (type=0) : Peuvent avoir degré élevé si beaucoup d'investissements reçus
- Investors (type=1) : Peuvent avoir degré élevé si investissent dans beaucoup de startups

**Le degré ne dépend PAS du type de nœud**, seulement du nombre d'interactions.

### 3. Training Set Uniquement

⚠️ **CRITIQUE** : Le degré est calculé **UNIQUEMENT sur le training set** !

```
Training split (70%):  ← Utilisé pour build_degree_dict()
Validation split (15%): Ignoré pour le degré
Test split (15%):      Ignoré pour le degré
```

**Pourquoi ?**

Si on incluait validation/test dans le calcul du degré → **Data leakage** !

Le modèle aurait accès à de l'information sur les edges futurs, ce qui biaise l'évaluation.

### 4. Nœuds Absents du Training

Si un nœud apparaît dans validation/test mais pas dans training :

```python
degree_tensor[new_node_id] = 0.0  # Initialisé à 0

# Dans HAR Loss:
degree_clamped = max(degree, 1.0)  # Clamped à 1 minimum
w = degree_clamped^(-0.5) = 1.0^(-0.5) = 1.0  # Poids maximal
```

**Interprétation :** Les nœuds nouveaux/rares reçoivent le **poids maximum**, ce qui est cohérent avec l'objectif de HAR (favoriser low-degree).

---

## Comparaison avec Autres Définitions de Degré

### Degré dans TGN vs HAR

| Concept | TGN (Mémoire Dynamique) | HAR Loss (Reweighting) |
|---------|-------------------------|------------------------|
| **Définition** | Nombre d'interactions **jusqu'au timestep t** | Nombre total d'interactions **dans training set** |
| **Évolution** | Dynamique (change à chaque timestep) | Statique (fixé au début training) |
| **Usage** | Mise à jour mémoire, agrégation messages | Reweighting de la loss |
| **Scope** | Local (par timestep) | Global (sur tout le dataset) |

### Degré Statique vs Dynamique

**HAR utilise degré STATIQUE** :
```python
# Calculé une seule fois au début du training
degree_dict = build_degree_dict(train_data)

# Reste constant pendant tout l'entraînement
# degree(StartupX) = 1 (même si le modèle prédit d'autres liens)
```

**Pourquoi statique ?**
- Simplicité d'implémentation
- Représente la "vraie popularité" observée dans les données
- Évite la complexité d'un degré dynamique

---

## Exemple Complet : Calcul HAR pour un Batch

### Batch de 3 Paires

```python
# Batch training
sources = [46, 42, 43]      # StartupX, DeepMind, OpenAI
destinations = [106, 103, 105]  # Angel, Y Comb, Sequoia

# Lookup degrés
src_degrees = degree_tensor[sources]
# → [1.0, 3.0, 2.0]

dst_degrees = degree_tensor[destinations]
# → [1.0, 3.0, 2.0]

# Calcul poids HAR (alpha = 0.5)
w_src = src_degrees ** (-0.5)
# → [1.000, 0.577, 0.707]

w_dst = dst_degrees ** (-0.5)
# → [1.000, 0.577, 0.707]

w_pair = w_src * w_dst
# → [1.000, 0.333, 0.500]
```

### Interprétation

| Paire | Degré Src | Degré Dst | Poids HAR | Interprétation |
|-------|-----------|-----------|-----------|----------------|
| StartupX → Angel | 1 | 1 | **1.000** | Deux nœuds rares → Poids MAXIMAL |
| DeepMind → Y Comb | 3 | 3 | **0.333** | Deux nœuds populaires → Poids RÉDUIT |
| OpenAI → Sequoia | 2 | 2 | **0.500** | Popularité moyenne → Poids modéré |

**Effet :** StartupX contribue 3× plus à la loss que DeepMind, forçant le modèle à apprendre sur les cas low-degree.

---

## Résumé Formule

### Degré

```
degree(node_i) = Σ [1 si node_i apparaît dans edge_j, 0 sinon]
                 j ∈ training_edges

              = nombre d'apparitions de node_i (comme src OU dst)
                dans le training set
```

### Poids HAR

```
w(node_i) = degree(node_i)^(-α)

où α > 0 (typiquement α = 0.5)
```

### Poids pour une Paire

```
w_pair(src, dst) = w(src) × w(dst)
                 = degree(src)^(-α) × degree(dst)^(-α)
                 = [degree(src) × degree(dst)]^(-α)
```

### Loss HAR Finale

```
L_HAR = Σ w_pair(src_i, dst_i) × L_contrastive(src_i, dst_i)
        i=1..N

où L_contrastive = InfoNCE style loss
```

---

## Vérification du Degré

### Commande pour Inspecter

Tu peux ajouter ce code pour vérifier les degrés calculés :

```python
# Après build_degree_dict()
degree_dict = build_degree_dict(train_data)

# Statistiques
degrees = list(degree_dict.values())
print(f"Nombre de nœuds: {len(degree_dict)}")
print(f"Degré min: {min(degrees)}")
print(f"Degré max: {max(degrees)}")
print(f"Degré moyen: {np.mean(degrees):.2f}")
print(f"Degré médian: {np.median(degrees):.2f}")

# Distribution
low_degree = sum(1 for d in degrees if d <= 5)
high_degree = sum(1 for d in degrees if d >= 20)
print(f"Low-degree (≤5): {low_degree} nœuds ({100*low_degree/len(degrees):.1f}%)")
print(f"High-degree (≥20): {high_degree} nœuds ({100*high_degree/len(degrees):.1f}%)")

# Top-10 nœuds populaires
top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop-10 nœuds les plus populaires:")
for node_id, degree in top_nodes:
    print(f"  Node {node_id}: degree={degree}")
```

---

## Conclusion

**Degré dans HAR Loss** :
1. ✅ Calculé en comptant les apparitions dans le **training set** uniquement
2. ✅ Mode **non-dirigé** (source ET destination comptent)
3. ✅ **Statique** (fixé au début, ne change pas durant training)
4. ✅ Stocké dans un **tensor** pour lookup rapide (O(1))
5. ✅ Utilisé pour calculer `w = degree^(-α)` (low-degree → poids élevé)

**Impact** : HAR force le modèle à apprendre sur les **paires rares** (low-degree), compensant le bias naturel vers les nœuds populaires (high-degree) du dataset.
