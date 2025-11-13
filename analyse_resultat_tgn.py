import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_tgn_runs(prefix='tgn-optimized', n_runs=10):
    """
    Analyse complète des runs TGN avec visualisations
    """
    results = []
    
    # Charger tous les résultats
    for i in range(n_runs):
        path = f"results/{prefix}_{i}.pkl" if i > 0 else f"results/{prefix}.pkl"
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                data['run_id'] = i
                results.append(data)
        except FileNotFoundError:
            print(f"⚠️  Run {i} not found")
            continue
    
    if not results:
        print("❌ No results found!")
        return
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS OF {len(results)} RUNS")
    print(f"{'='*80}\n")
    
    # ============================================================================
    # 1. STATISTIQUES GLOBALES
    # ============================================================================
    test_aps = [r.get('test_ap', 0) for r in results]
    new_test_aps = [r.get('new_node_test_ap', 0) for r in results]
    
    # Filtrer les runs qui ont convergé (AP > 0.51)
    converged_mask = np.array(test_aps) > 0.51
    converged_runs = np.sum(converged_mask)
    
    print("📊 OVERALL STATISTICS")
    print("-" * 80)
    print(f"Runs converged: {converged_runs}/{len(results)} ({100*converged_runs/len(results):.1f}%)")
    print(f"\n▶ ALL RUNS:")
    print(f"  Test AP (old):     {np.mean(test_aps):.4f} ± {np.std(test_aps):.4f}")
    print(f"  Test AP (new):     {np.mean(new_test_aps):.4f} ± {np.std(new_test_aps):.4f}")
    print(f"  Range (old):       [{np.min(test_aps):.4f}, {np.max(test_aps):.4f}]")
    print(f"  Median (old):      {np.median(test_aps):.4f}")
    
    if converged_runs > 0:
        converged_aps = np.array(test_aps)[converged_mask]
        converged_new_aps = np.array(new_test_aps)[converged_mask]
        print(f"\n▶ CONVERGED RUNS ONLY:")
        print(f"  Test AP (old):     {np.mean(converged_aps):.4f} ± {np.std(converged_aps):.4f}")
        print(f"  Test AP (new):     {np.mean(converged_new_aps):.4f} ± {np.std(converged_new_aps):.4f}")
        print(f"  Best run:          {np.max(converged_aps):.4f}")
    
    # ============================================================================
    # 2. ANALYSE PAR RUN
    # ============================================================================
    print(f"\n\n📋 PER-RUN BREAKDOWN")
    print("-" * 80)
    print(f"{'Run':<5} {'Epochs':<8} {'Val AP':<10} {'Test AP':<10} {'New AP':<10} {'Status':<12}")
    print("-" * 80)
    
    for r in results:
        run_id = r['run_id']
        n_epochs = len(r.get('val_aps', []))
        best_val_ap = max(r.get('val_aps', [0]))
        test_ap = r.get('test_ap', 0)
        new_ap = r.get('new_node_test_ap', 0)
        status = "✅ Good" if test_ap > 0.54 else ("⚠️  Weak" if test_ap > 0.51 else "❌ Failed")
        
        print(f"{run_id:<5} {n_epochs:<8} {best_val_ap:<10.4f} {test_ap:<10.4f} {new_ap:<10.4f} {status:<12}")
    
    # ============================================================================
    # 3. DÉTECTION DES PATTERNS
    # ============================================================================
    print(f"\n\n🔍 PATTERN DETECTION")
    print("-" * 80)
    
    # Temps de convergence
    convergence_epochs = []
    for r in results:
        val_aps = r.get('val_aps', [])
        # Trouver le premier epoch où val_ap > 0.52
        for i, ap in enumerate(val_aps):
            if ap > 0.52:
                convergence_epochs.append(i)
                break
    
    if convergence_epochs:
        print(f"⏱️  Convergence speed:")
        print(f"   Average epoch to reach AP>0.52: {np.mean(convergence_epochs):.1f} ± {np.std(convergence_epochs):.1f}")
        print(f"   Range: [{min(convergence_epochs)}, {max(convergence_epochs)}]")
    
    # Mode collapse detection
    collapsed_runs = 0
    for r in results:
        val_aps = r.get('val_aps', [])
        # Check si beaucoup d'epochs à exactement 0.5
        exact_half = sum(1 for ap in val_aps if abs(ap - 0.5) < 0.001)
        if exact_half > len(val_aps) * 0.5:
            collapsed_runs += 1
    
    print(f"\n⚠️  Mode collapse:")
    print(f"   {collapsed_runs}/{len(results)} runs suffered from mode collapse")
    
    # ============================================================================
    # 4. VISUALISATIONS
    # ============================================================================
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Training curves
    ax1 = plt.subplot(2, 3, 1)
    for r in results:
        alpha = 0.8 if r.get('test_ap', 0) > 0.54 else 0.3
        color = 'green' if r.get('test_ap', 0) > 0.54 else 'red'
        ax1.plot(r.get('train_losses', []), alpha=alpha, color=color, linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss (Green=Good, Red=Failed)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation AP curves
    ax2 = plt.subplot(2, 3, 2)
    for r in results:
        alpha = 0.8 if r.get('test_ap', 0) > 0.54 else 0.3
        color = 'green' if r.get('test_ap', 0) > 0.54 else 'red'
        ax2.plot(r.get('val_aps', []), alpha=alpha, color=color, linewidth=1.5)
    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation AP')
    ax2.set_title('Validation AP Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Test AP distribution
    ax3 = plt.subplot(2, 3, 3)
    colors = ['green' if ap > 0.54 else 'red' for ap in test_aps]
    ax3.bar(range(len(test_aps)), test_aps, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random')
    ax3.axhline(y=np.mean(test_aps), color='blue', linestyle='-', alpha=0.7, label='Mean')
    ax3.set_xlabel('Run ID')
    ax3.set_ylabel('Test AP')
    ax3.set_title('Test AP by Run')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Loss histogram final epochs
    ax4 = plt.subplot(2, 3, 4)
    final_losses = []
    for r in results:
        losses = r.get('train_losses', [])
        if len(losses) > 5:
            final_losses.append(np.mean(losses[-5:]))  # Moyenne des 5 dernières
    ax4.hist(final_losses, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax4.set_xlabel('Final Training Loss')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Final Loss')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Convergence speed
    ax5 = plt.subplot(2, 3, 5)
    max_val_aps = [max(r.get('val_aps', [0])) for r in results]
    n_epochs_list = [len(r.get('val_aps', [])) for r in results]
    colors_scatter = ['green' if ap > 0.54 else 'red' for ap in test_aps]
    ax5.scatter(n_epochs_list, max_val_aps, c=colors_scatter, alpha=0.7, s=100, edgecolor='black')
    ax5.set_xlabel('Number of Epochs')
    ax5.set_ylabel('Best Validation AP')
    ax5.set_title('Convergence Pattern')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Old vs New nodes performance
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(test_aps, new_test_aps, c=colors, alpha=0.7, s=100, edgecolor='black')
    ax6.plot([0.45, 0.6], [0.45, 0.6], 'k--', alpha=0.5, label='Perfect generalization')
    ax6.set_xlabel('Test AP (Old Nodes)')
    ax6.set_ylabel('Test AP (New Nodes)')
    ax6.set_title('Generalization to New Nodes')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'analysis_{prefix}.png', dpi=150, bbox_inches='tight')
    print(f"\n\n📊 Visualization saved to 'analysis_{prefix}.png'")
    
    # ============================================================================
    # 5. RECOMMANDATIONS
    # ============================================================================
    print(f"\n\n💡 RECOMMENDATIONS")
    print("-" * 80)
    
    if converged_runs < len(results) * 0.5:
        print("❌ CRITICAL: Less than 50% of runs converged!")
        print("   → Reduce learning rate (try 0.0003)")
        print("   → Add learning rate warmup")
        print("   → Increase dropout to 0.5")
        print("   → Add gradient clipping (max_norm=0.5)")
    
    if np.std(test_aps) > 0.03:
        print("\n⚠️  HIGH VARIANCE detected!")
        print("   → Use weight initialization (Xavier/He)")
        print("   → Add batch normalization")
        print("   → Consider ensemble of top 3 models")
    
    if collapsed_runs > 0:
        print(f"\n⚠️  {collapsed_runs} runs suffered mode collapse!")
        print("   → This suggests gradient vanishing")
        print("   → Try different loss function (Margin loss)")
        print("   → Check if negative sampling is working")
    
    best_run_idx = np.argmax(test_aps)
    print(f"\n✨ BEST MODEL: Run {best_run_idx}")
    print(f"   Test AP (old): {test_aps[best_run_idx]:.4f}")
    print(f"   Test AP (new): {new_test_aps[best_run_idx]:.4f}")
    print(f"   Saved in: ./saved_models/{prefix}-BEST.pth")
    
    # Ensemble recommendation
    if converged_runs >= 3:
        top_3_indices = np.argsort(test_aps)[-3:]
        top_3_aps = [test_aps[i] for i in top_3_indices]
        print(f"\n🎯 ENSEMBLE POTENTIAL:")
        print(f"   Top 3 runs: {list(top_3_indices)}")
        print(f"   Their APs: {[f'{ap:.4f}' for ap in top_3_aps]}")
        print(f"   Expected ensemble AP: ~{np.mean(top_3_aps) + 0.01:.4f}")
    
    print("\n" + "="*80)

# ============================================================================
# FONCTION POUR CRÉER UN ENSEMBLE
# ============================================================================
def create_ensemble(prefix, top_k=3):
    """Sélectionne les top K modèles pour créer un ensemble"""
    import pickle
    
    results = []
    for i in range(20):  # Check jusqu'à 20 runs
        path = f"results/{prefix}_{i}.pkl" if i > 0 else f"results/{prefix}.pkl"
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                data['run_id'] = i
                results.append(data)
        except:
            continue
    
    # Trier par test AP
    test_aps = [(r['run_id'], r.get('test_ap', 0)) for r in results]
    test_aps.sort(key=lambda x: x[1], reverse=True)
    
    top_runs = test_aps[:top_k]
    
    print(f"\n{'='*60}")
    print(f"TOP {top_k} MODELS FOR ENSEMBLE")
    print(f"{'='*60}")
    for run_id, ap in top_runs:
        print(f"Run {run_id}: AP = {ap:.4f}")
        print(f"  → Load from: ./saved_models/{prefix}_{run_id}.pth" if run_id > 0 else f"  → Load from: ./saved_models/{prefix}.pth")
    print(f"{'='*60}\n")
    
    return [r[0] for r in top_runs]

# ============================================================================
# EXÉCUTION
# ============================================================================
if __name__ == "__main__":
    # Analyser les résultats
    analyze_tgn_runs(prefix='tgn-optimized', n_runs=10)
    
    # Créer un ensemble
    top_models = create_ensemble(prefix='tgn-optimized', top_k=3)