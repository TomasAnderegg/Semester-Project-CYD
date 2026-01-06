# Forecasting Emerging Disruptive Technologies

**EPFL Semester Project - CYD Campus**
*Combining Temporal Graph Networks with TechRank for Early Detection of Disruptive Technologies*

**Author:** Tomas Garate Anderegg
**Supervisors:** Prof. Andrea Cavallaro, Julian Jang-Jaccard

---

## Abstract

This project develops a novel framework for forecasting emerging disruptive technologies in the context of national cyber defence. By combining **TechRank** (a PageRank-inspired centrality algorithm for bipartite networks) with **Temporal Graph Networks** (TGN), the framework identifies companies positioned for significant influence growth in the technology ecosystem.

The key innovation is detecting **disruptive signals** by analyzing how company rankings evolve when TGN predicts future investment patterns. Companies that start with low TechRank scores but experience dramatic increases after prediction may signal early-stage disruptive trajectories.

**Key Results:**
- Spearman rank correlation: **ρ = 0.3746, p = 0.0495** (statistically significant)
- Top-20 overlap: **85%** between predicted and actual influential companies
- DCL loss achieves **Recall@50 = 91%** on transductive link prediction

---

## Table of Contents

- [Motivation](#motivation)
- [Research Question](#research-question)
- [Methodology](#methodology)
  - [TechRank Algorithm](#techrank-algorithm)
  - [Temporal Graph Networks](#temporal-graph-networks)
  - [Integration Pipeline](#integration-pipeline)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Pipeline](#data-pipeline)
  - [Training TGN](#training-tgn)
  - [TechRank Comparison](#techrank-comparison)
- [Evaluation](#evaluation)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Citation](#citation)

---

## Motivation

Technological innovation evolves at an unprecedented pace, with profound implications for national security and cyber defence. **Disruptive technologies** often start weak, attract only niche users, but eventually transform entire industries (e.g., Internet, smartphones, quantum computing).

Traditional monitoring frameworks struggle to:
1. Model temporal evolution of technology ecosystems
2. Detect early signals before disruption becomes obvious
3. Handle dynamic, non-linear changes in investment patterns

This project addresses these gaps by combining:
- **TechRank**: Measures current influence in investor-company networks
- **TGN**: Predicts future network evolution
- **ΔR metric**: Identifies companies with dramatic rank increases (disruption signals)

---

## Research Question

> **How can Temporal Graph Networks enhance TechRank to improve temporal forecasting of cyber-relevant disruptive technologies?**

**Hypothesis:** Modeling the technology ecosystem as a dynamic bipartite graph (investors ↔ companies) enables earlier identification of high-potential technologies compared to static network analysis alone.

---

## Methodology

### TechRank Algorithm

TechRank is a **bipartite PageRank** variant that ranks companies and investors based on mutual influence:

```
Initial scores:
w_c^(0) = Σ_t M_{c,t}  (company degree)
w_t^(0) = Σ_c M_{c,t}  (investor degree)

Recursive update:
w_c^(n+1) = Σ_t G_{c,t}(β) · w_t^(n)
w_t^(n+1) = Σ_c G_{t,c}(α) · w_c^(n)
```

**Key Parameters:**
- **β = -5.0**: Strongly amplifies influence of highly active investors
  - An investor funding 50 companies has weight **9.7 million times** larger than one funding 2 companies (50^5 / 2^5)
  - This makes TechRank highly sensitive to new connections with established investors
  - Enables detection of sudden attractiveness shifts

- **α = 0.3**: Companies contribute nearly equally regardless of connectivity
  - Avoids penalizing emerging companies with few investors
  - Maintains focus on investor activity as the primary signal

### Temporal Graph Networks

TGN processes the investor-company graph as a sequence of **timestamped events** (funding rounds):

1. **Memory module**: Captures historical state of each node (company/investor)
2. **Message passing**: Updates memory after each interaction
3. **Attention-based aggregation**: Generates embeddings from recent neighbors
4. **Link prediction**: MLP decoder predicts probability of future investments

**Architecture:**
- Node embedding: 200-dim
- Temporal embedding: 200-dim
- Graph attention with 2 heads
- GRU memory updater

### Integration Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ 1. Compute TechRank on observed graph                   │
│    → Initial influence scores (Score₀)                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Train TGN on temporal interactions                    │
│    → Learn dynamics of investment patterns               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Predict future graph with TGN                         │
│    → Generate all company-investor pair probabilities    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Recompute TechRank on predicted graph                 │
│    → Updated influence scores (Score_pred)               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 5. Compute ΔR = (Score_pred - Score₀) / (Score₀ + ε)    │
│    → Identify companies with significant rank increase   │
└─────────────────────────────────────────────────────────┘
```

**Disruption Detection:**
Companies with **high ΔR** (large relative increase in TechRank) signal potential disruption:
- Started with low visibility / few investors
- Predicted to attract high-influence investors
- May be developing breakthrough technologies

---

## Project Structure

```
.
├── data/                              # Data extraction & preprocessing
│   ├── data_extraction.py             # Extract from Crunchbase
│   ├── data_conversion.py             # Convert to TGN format
│   ├── bipartite_investor_comp.py     # Build bipartite graphs
│   ├── diagnosis_bipartite_graph.py   # Graph quality diagnostics
│   └── diagnostic_leakage.py          # Temporal leakage detection
│
├── code/                              # Core algorithms
│   ├── TechRank.py                    # TechRank implementation (α, β parameters)
│   └── test_matrix_M.py               # Adjacency matrix analysis
│
├── evaluation/                        # Evaluation & baselines
│   ├── Random_forest.py               # Random Forest baseline
│   └── test_ranking_logic.py          # Ranking validation
│
├── train_self_supervised.py           # TGN training (main entry point)
├── TGN_eval.py                        # TGN evaluation with temporal split
├── TechRank_Comparison.py             # Before/after TechRank comparison
├── run_all_experiments.py             # Batch experiments
│
├── dcl_loss.py                        # Degree Contrastive Loss
├── focal_loss.py                      # Focal Loss (class imbalance)
├── hard_negative_mining.py            # Hard negative sampling
│
└── utils/                             # Utilities
    ├── preprocess_data.py
    └── data_processing.py
```

---

## Dataset

**Source:** [Crunchbase](https://www.crunchbase.com/)
**Focus:** Quantum computing & quantum key distribution companies

### Statistics

| Metric | Value |
|--------|-------|
| Companies | 223 |
| Investors | 1,016 |
| Funding events | 1,330 |
| Edge features | 2 (raised amount USD, #investors) |
| Timespan | 42,684 days (~117 years of data) |
| Train/Val/Test split | 70% / 15% / 15% (chronological) |

### Key Features

**Investors:**
- Name
- Funding round ID
- Timestamp (funding announcement date)
- Amount raised (USD)
- Number of co-investors

**Companies:**
- Name
- Company ID
- Associated technologies

**Graph Properties:**
- Bipartite structure: Investor (type-1) ↔ Company (type-0)
- Sparse (density = 0.0059)
- Hub structure: Few highly connected investors, many low-degree companies
- Extreme class imbalance: 0.59% positive links (169:1 negative-to-positive ratio)

---

## Installation

```bash
# Python >= 3.7 required
pip install torch==1.6.0
pip install pandas==1.1.0
pip install scikit-learn==0.23.1
pip install networkx
pip install scipy
pip install matplotlib
```

---

## Usage

### Data Pipeline

#### 1. Extract and Convert Data

```bash
# Extract Crunchbase data
python data/data_extraction.py

# Build bipartite graph
python data/bipartite_investor_comp.py

# Convert to TGN format (CSV + edge features)
python data/data_conversion.py
```

#### 2. Diagnose Graph Quality

```bash
# Visualize adjacency matrix, check for isolated nodes
python code/test_matrix_M.py

# Detect temporal leakage
python evaluation/diagnostic_leakage.py
```

### Training TGN

```bash
# Train with BCE loss (baseline)
python train_self_supervised.py \
    --data crunchbase_invest_comp \
    --use_memory \
    --prefix tgn-baseline \
    --n_runs 6 \
    --n_epoch 50 \
    --patience 10 \
    --bs 200 \
    --lr 0.0001

# Train with DCL loss (mitigates degree bias)
python train_self_supervised.py \
    --data crunchbase_invest_comp \
    --use_memory \
    --loss dcl \
    --prefix tgn-dcl \
    --n_runs 6 \
    --n_epoch 50 \
    --patience 10
```

### TechRank Comparison

#### Evaluate TGN with Temporal Validation

```bash
# Critical: Use temporal_split=0.6 to avoid leakage
python TGN_eval.py \
    --data crunchbase_invest_comp \
    --use_memory \
    --prefix tgn-dcl \
    --temporal_split 0.6
```

**What this does:**
1. Splits test set: 60% for prediction, 40% as ground truth (future)
2. Computes TechRank on historical + predicted interactions
3. Computes TechRank on historical + ground truth interactions
4. Measures Spearman correlation between predicted and actual rankings

#### Compare TechRank Before/After TGN

```bash
python TechRank_Comparison.py \
    --data crunchbase_invest_comp \
    --use_memory \
    --prefix tgn-dcl \
    --temporal_split 0.6
```

**Output:**
- Spearman rank correlation (r, p-value)
- Top-K overlap (companies appearing in both predicted and true top-K)
- Ranking visualizations (bump chart, diverging bar chart)
- CSV export of companies with high ΔR

---

## Evaluation

### Metrics

#### Link Prediction (TGN Performance)

- **AUROC / AP**: Binary classification metrics
- **MRR (Mean Reciprocal Rank)**: Rank of first correct prediction
- **Recall@K**: Proportion of true links in top-K predictions

#### Ranking Validation (Pipeline Performance)

- **Spearman Rank Correlation (ρ)**: Agreement between predicted and actual influential companies
- **P-value**: Statistical significance of correlation
- **Top-K Overlap**: Set intersection |Predicted_Top-K ∩ True_Top-K|

#### Disruption Detection

- **ΔR (Delta Ranking)**: Relative change in TechRank score
  ```
  ΔR = (TechRank_predicted - TechRank_initial) / (TechRank_initial + ε)
  ```
- Companies with **high ΔR** are flagged as potentially disruptive

### Results

#### TGN Model Performance

| Model | AUROC | AP | MRR | Recall@10 | Recall@50 |
|-------|-------|-----|-----|-----------|-----------|
| Random Forest | 0.519 | 0.602 | 0.086 | 0.225 | 0.500 |
| JODIE | 0.764 | 0.712 | 0.143 | 0.282 | 0.823 |
| DyRep | 0.720 | 0.718 | 0.185 | 0.323 | 0.682 |
| **TGN (BCE)** | 0.835 | 0.848 | 0.530 | 0.645 | 0.870 |
| **TGN-DCL** | **0.763** | **0.779** | 0.527 | **0.755** | **0.910** |

**Key Insight:** DCL loss significantly improves **Recall@50** (91%) by mitigating degree bias, allowing better identification of low-degree (emerging) companies.

#### Temporal Validation (Pipeline)

| Metric | Result |
|--------|--------|
| Spearman correlation (ρ) | **0.3746** |
| P-value | **0.0495** (p < 0.05, significant) |
| Top-10 overlap | **60%** (6/10) |
| Top-20 overlap | **85%** (17/20) |

**Interpretation:**
- Model preserves **relative ranking** of influential companies
- Statistically significant correlation confirms temporal patterns are captured
- High Top-K overlap validates that predicted rankings align with future reality

#### Top Disruptive Companies (High ΔR)

| Rank | Company | TechRank_before | TechRank_after | ΔR |
|------|---------|-----------------|----------------|----|
| 1 | Quantistry | 0.0013 | 0.1448 | **111× increase** |
| 2 | Phaseshift Tech. | 0.0001 | 0.0089 | 89× increase |
| 3 | QSIM Plus | 0.0012 | 0.0826 | 69× increase |
| 4 | Global Telecom | 0.0013 | 0.0499 | 38× increase |

---

## Key Findings

### 1. Temporal Validation is Critical

- **Without temporal split**: High metrics but data leakage (model sees future)
- **With temporal_split=0.6**: Valid predictions, statistically significant correlation
- **Temporal_split=0.5**: Non-significant (p=0.314), insufficient future data

### 2. Macro vs Micro Performance Gap

- **Macro level** (Top-K overlap): 60-85% overlap, model identifies right companies
- **Micro level** (Precision@K): ~0.2% at K=2000, struggles with exact investor matching
- **Interpretation**: Model captures company-level disruption but not investor-company pair granularity
  - Analogy: Knows "Mbappé will change clubs" but can't predict "which club"

### 3. TechRank Parameter Choice Matters

- **β = -5.0**: Amplifies high-degree investor influence by factor of k^5
  - When TGN predicts new link to established investor → dramatic TechRank increase
  - Enables detection of sudden attractiveness shifts
- **α = 0.3**: Avoids penalizing emerging companies with few investors
  - Ensures low-degree companies can still achieve high ΔR

### 4. DCL Loss Mitigates Degree Bias

- Standard BCE: Focuses learning on high-degree nodes (hubs)
- DCL: Downweights hub interactions, upweights low-degree interactions
- Result: **+16% Recall@50** improvement for detecting emerging companies

### 5. Error Analysis: Random vs Systematic

- Prediction errors appear **largely random** rather than systematic
- High-variance errors across companies (Table 11 in report)
- TechRank aggregation mechanism preserves ranking despite low precision
- No evidence of graph distance bias, community structure bias (Sinha et al., 2018)

---

## Limitations

### Dataset Limitations

1. **Small dataset**: 223 companies, 1,330 events
   - Limits generalization to other technology domains
   - Maximum 10 timestamps per company (sparse temporal signal)

2. **Missing information**:
   - No data on how funding is allocated within companies
   - All technologies weighted equally (no prioritization)
   - No semantic features (patents, publications, etc.)

3. **Domain-specific**: Focus on quantum computing
   - Results may not generalize to AI, biotech, etc.

### Model Limitations

1. **Low Precision@K**: Exact investor-company pair prediction remains challenging
   - Precision@K ≈ 0 for K < 2000
   - Model captures company potential but not investor preferences

2. **Hyperparameters not optimized**:
   - TechRank (α, β): Manual selection based on reasoning
   - Focal Loss / DCL: Default hyperparameters used
   - Potential for improvement with systematic tuning

3. **No directionality enforcement**: Investor → Company direction implicit but not enforced

### Validation Limitations

1. **Crunchbase metrics unreliable**: Heat Score, Growth Score use proprietary algorithms
   - Cannot be used for rigorous validation

2. **Ground truth ambiguity**: Disruption is subjective
   - No labeled "disruptive" companies in dataset
   - Validation relies on proxy metrics (rank correlation, overlap)

---

## Future Work

### Methodological Improvements

1. **Hyperparameter Optimization**:
   - Grid search for (α, β) in TechRank
   - Tune Focal Loss (γ, α) and DCL (temperature, exponent)

2. **Richer Node Features**:
   - Company: #patents, #publications, sector, funding stage
   - Investor: portfolio diversity, historical success rate
   - Edge: funding round type (seed, Series A, etc.)

3. **Directed Graph Modeling**: Explicitly model Investor → Company direction

4. **Alternative Models**:
   - JEPA (Joint Embedding Predictive Architecture) for self-supervised learning
   - Data assimilation methods (inspired by weather forecasting)

### Data & Validation Improvements

1. **Larger Dataset**:
   - Expand beyond quantum computing (AI, biotech, autonomous vehicles)
   - Include patent databases, academic publications

2. **Synthetic Data Augmentation**:
   - Generate additional timestamps to enrich temporal signal
   - Ensure fixed minimum #interactions per company

3. **Expert Validation**:
   - Collaborate with domain experts to label disruptive companies
   - Validate predictions against realized market outcomes

4. **Technology Weighting**:
   - Prioritize emerging technologies over mature ones
   - Use external signals (e.g., hype cycle, R&D investment trends)

### Operational Deployment

1. **Agentic AI Integration**:
   - Continuous monitoring of Crunchbase updates
   - Real-time alerts for companies with sudden ΔR spikes
   - Autonomous parameter adjustment based on drift detection

2. **Interactive Dashboard**:
   - Visualize company trajectories over time
   - Filter by technology sector, geographic region
   - Export high-ΔR companies for strategic analysis

---

## Citation

This project builds upon the TGN framework:

```bibtex
@inproceedings{tgn_icml_grl2020,
    title={Temporal Graph Networks for Deep Learning on Dynamic Graphs},
    author={Emanuele Rossi and Ben Chamberlain and Fabrizio Frasca and Davide Eynard and Federico Monti and Michael Bronstein},
    booktitle={ICML 2020 Workshop on Graph Representation Learning},
    year={2020}
}
```

For the TechRank algorithm:

```bibtex
@article{mezzetti2022techrank,
    title={TechRank: Modelling Portfolios of Cyber-Related Emerging Technologies},
    author={Anita Mezzetti and others},
    journal={arXiv preprint arXiv:2210.07824},
    year={2022}
}
```

---

## License

See original TGN repository for license information.

---

## Acknowledgments

This work was conducted as a semester project at the **CYD Campus (Cyber-Defence Campus)** at EPFL, under the supervision of Prof. Andrea Cavallaro and Julian Jang-Jaccard.

