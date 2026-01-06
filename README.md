# TGN for Investment Prediction with TechRank Validation

This project applies Temporal Graph Networks (TGN) to predict future investment links in the Crunchbase dataset, using TechRank as a validation mechanism to assess prediction quality.

## Project Overview

The project models investor-company relationships as a dynamic bipartite graph where:
- **Nodes**: Companies (bipartite=0) and Investors (bipartite=1)
- **Edges**: Investment events with timestamps
- **Task**: Predict future investment links using TGN
- **Validation**: Compare TechRank scores before/after TGN predictions to assess if predictions align with PageRank-based company quality

### Key Innovation

Rather than relying solely on traditional metrics (Precision@K, Recall@K), we validate TGN predictions by checking if they improve TechRank scores - a PageRank-inspired algorithm that ranks companies based on their investor network quality.

## Project Structure

```
.
├── data/                              # Data processing scripts
│   ├── data_extraction.py             # Extract raw data from Crunchbase
│   ├── data_conversion.py             # Convert to TGN format
│   ├── data_investment_conversion.py  # Investment-specific conversions
│   ├── bipartite_investor_comp.py     # Build bipartite graphs
│   ├── bipartite_tech_comp.py         # Technology-company graphs
│   ├── diagnosis_bipartite_graph.py   # Graph diagnostics
│   └── diagnosis_crunchbase_tgn.py    # TGN-specific diagnostics
│
├── code/                              # Core algorithms
│   ├── TechRank.py                    # TechRank algorithm implementation
│   └── test_matrix_M.py               # Adjacency matrix analysis
│
├── evaluation/                        # Evaluation tools
│   ├── Random_forest.py               # Random Forest baselines
│   ├── test_ranking_logic.py          # Ranking logic tests
│   └── diagnostic_leakage.py          # Data leakage detection
│
├── train_self_supervised.py           # Main TGN training script
├── TGN_eval.py                        # TGN evaluation with temporal validation
├── TechRank_Comparison.py             # Compare TechRank before/after TGN
├── run_all_experiments.py             # Batch experiment runner
├── compare_models_results.py          # Cross-model comparison
│
├── dcl_loss.py                        # Dual Contrastive Loss
├── focal_loss.py                      # Focal Loss for imbalanced data
├── hard_negative_mining.py            # Hard negative sampling
├── plot_loss_comparison.py            # Training loss visualization
│
└── utils/                             # Utility modules
    ├── preprocess_data.py             # Data preprocessing
    ├── data_processing.py             # Processing utilities
    └── utils.py                       # General utilities
```

## Requirements

```bash
python >= 3.7
pandas==1.1.0
torch==1.6.0
scikit_learn==0.23.1
networkx
numpy
matplotlib
scipy
```

## Data Pipeline

### 1. Data Extraction and Conversion

```bash
# Extract Crunchbase data
python data/data_extraction.py

# Convert to bipartite graph format
python data/bipartite_investor_comp.py

# Convert to TGN format (edge list with timestamps)
python data/data_conversion.py
```

### 2. Data Diagnostics

```bash
# Analyze graph structure and adjacency matrix
python code/test_matrix_M.py

# Check for data leakage
python evaluation/diagnostic_leakage.py

# Diagnose TGN data format
python data/diagnosis_crunchbase_tgn.py
```

## TechRank Algorithm

TechRank is a PageRank-inspired algorithm for bipartite graphs with two key parameters:

- **β**: Controls investor degree weighting
  - Low β (e.g., 0.5): Rewards backing by highly connected investors
  - High β (e.g., 1.5): Penalizes over-diversified investors

- **α**: Controls company degree weighting
  - Low α: Rewards companies with many investors
  - High α: Penalizes over-funded companies

### Running TechRank

```python
from code.TechRank import TechRank

# Initialize with bipartite graph
techrank = TechRank(bipartite_graph)

# Compute rankings
rankings = techrank.compute_techrank(alpha=0.5, beta=0.5)
```

## Model Training

### Self-Supervised Training

```bash
# Train TGN with default parameters
python train_self_supervised.py \
    --data crunchbase_invest_comp \
    --use_memory \
    --prefix tgn-investment \
    --n_runs 6 \
    --n_epoch 50 \
    --patience 10 \
    --bs 200 \
    --lr 0.0001
```

### Key Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--data` | crunchbase_invest_comp | Dataset name |
| `--use_memory` | flag | Enable memory module |
| `--n_runs` | 6 | Number of training runs |
| `--n_epoch` | 50 | Maximum epochs |
| `--patience` | 10 | Early stopping patience |
| `--bs` | 200 | Batch size |
| `--lr` | 0.0001 | Learning rate |
| `--n_degree` | 10 | Neighbors to sample |
| `--n_head` | 2 | Attention heads |
| `--n_layer` | 1 | Graph attention layers |

### Loss Functions

The project supports multiple loss functions:

- **Binary Cross-Entropy** (default): Standard link prediction loss
- **Focal Loss**: Handles class imbalance by focusing on hard examples
- **DCL Loss**: Dual Contrastive Loss for better negative sampling

```bash
# Train with Focal Loss
python train_self_supervised.py --loss focal --focal_alpha 0.25 --focal_gamma 2.0

# Train with DCL Loss
python train_self_supervised.py --loss dcl
```

## Evaluation

### TGN Evaluation with Temporal Validation

The evaluation uses temporal splitting to avoid data leakage:

```bash
# Evaluate with 60% temporal split (critical for avoiding leakage)
python TGN_eval.py \
    --data crunchbase_invest_comp \
    --use_memory \
    --prefix tgn-investment \
    --temporal_split 0.6
```

**Key Metrics:**
- **Precision@K**: Fraction of top-K predictions that are correct
- **Top-K Overlap**: Set intersection between predicted and true top-K companies
- **P-value**: Statistical significance of overlap vs random baseline

**Temporal Split Rationale:**
- Without temporal split: Risk of data leakage (evaluating on past interactions)
- With `temporal_split=0.6`: Only evaluate on last 40% of test interactions
- This ensures predictions are truly future-oriented

### TechRank Validation

Compare TechRank rankings before and after adding TGN predictions:

```bash
python TechRank_Comparison.py \
    --data crunchbase_invest_comp \
    --use_memory \
    --prefix tgn-investment \
    --temporal_split 0.6
```

**Analysis Performed:**
- Spearman rank correlation between TechRank_before and TechRank_after
- Company-level delta analysis (which companies gain/lose rank)
- Statistical significance testing
- Visualization of ranking changes

**Interpretation:**
- High correlation + low p-value → TGN predictions align with graph structure
- Top companies gaining rank → Model correctly identifies quality investment targets
- Random correlation → Model predictions may not respect network topology

### Example Results

```
Top-20 Overlap Analysis:
   Predicted companies: 20
   True future partners: 150
   Overlap: 17 companies
   Overlap percentage: 85.0%
   P-value: 0.049

Spearman Rank Correlation:
   Correlation: 0.8542
   P-value: 0.00001
   Significance: ***
```

## Analysis Tools

### Graph Structure Analysis

```bash
# Analyze adjacency matrix and graph properties
python code/test_matrix_M.py
```

Output:
- Node counts (companies/investors)
- Degree distributions
- Density metrics
- Connected components
- Matrix visualizations (raw and sorted by degree)
- TechRank readiness assessment

### Prediction Bias Analysis

The evaluation includes scatter plot analysis of prediction errors per company:

```python
# Automatically generated during TGN_eval.py with --bias_analysis flag
python TGN_eval.py --bias_analysis --temporal_split 0.6
```

Generates:
- Scatter plot: true positives vs false positives per company
- R² trend line to identify systematic patterns
- CSV export of per-company error rates
- Top-5 false positive companies for error analysis

### Loss Comparison

```bash
# Compare training losses across different runs
python plot_loss_comparison.py
```

## Utility Scripts

### Emoji Removal

Removes emojis from print statements (for Windows console compatibility):

```bash
python remove_emojis.py
```

### Batch Experiments

```bash
# Run multiple experiments with different hyperparameters
python run_all_experiments.py
```

### Model Comparison

```bash
# Compare results across different models
python compare_models_results.py
```

## Key Findings

1. **Temporal Validation is Critical**
   - Without temporal split: High Precision@K but data leakage
   - With temporal_split=0.6: Lower Precision@K but valid predictions

2. **Macro vs Micro Performance**
   - Macro level (Top-K overlap): 60-85% overlap, statistically significant
   - Micro level (Precision@K): ~1-3%, indicates granularity mismatch
   - Interpretation: Model identifies right companies but struggles with exact investor matching

3. **TechRank Validation**
   - High Spearman correlation (>0.8) validates that predictions respect graph topology
   - Companies gaining TechRank after TGN predictions confirms quality alignment

4. **Metrics Interpretation**
   - **Recall@K**: Used during training with negative sampling (1 true + N false)
   - **Precision@K**: Used in evaluation with full candidate space (all possible pairs)
   - Both are valid for different evaluation contexts

## Citation

This project builds on the TGN framework:

```bibtex
@inproceedings{tgn_icml_grl2020,
    title={Temporal Graph Networks for Deep Learning on Dynamic Graphs},
    author={Emanuele Rossi and Ben Chamberlain and Fabrizio Frasca and Davide Eynard and Federico Monti and Michael Bronstein},
    booktitle={ICML 2020 Workshop on Graph Representation Learning},
    year={2020}
}
```

## License

See original TGN repository for license information.
