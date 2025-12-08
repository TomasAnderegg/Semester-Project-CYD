import pandas as pd
import pickle
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import skew, kurtosis
import warnings
from networkx.algorithms.bipartite import is_bipartite
from typing import List, Dict, Tuple, Any
warnings.filterwarnings('ignore')

# ===================================================================
# CONFIGURATION
# ===================================================================

NUM_COMP = 10000
NUM_TECH = 10000
FLAG_CYBERSECURITY = True

SAVE_DIR_CLASSES = "savings/bipartite_invest_comp/classes"
SAVE_DIR_NETWORKS = "savings/bipartite_invest_comp/networks"
SAVE_DIR_ANALYSIS = "analysis/graph_quality"

# ===================================================================
# FONCTIONS DE CALCUL DES MÉTRIQUES
# ===================================================================

def gini_coefficient(values: np.ndarray) -> float:
    """Coefficient de Gini (0=égalité parfaite, 1=max inégalité)"""
    values = np.sort(values)
    n = len(values)
    if n == 0:
        return 0
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))

def calculate_degree_statistics(G, investors: List, companies: List) -> Dict:
    """Calcule les statistiques de degré pour les deux partitions"""
    deg_inv = [G.degree(i) for i in investors]
    deg_comp = [G.degree(c) for c in companies]
    
    return {
        'investors': {
            'degrees': deg_inv,
            'mean': np.mean(deg_inv),
            'median': np.median(deg_inv),
            'std': np.std(deg_inv),
            'max': np.max(deg_inv),
            'min': np.min(deg_inv),
            'skewness': skew(deg_inv) if len(deg_inv) > 1 else 0,
            'kurtosis': kurtosis(deg_inv) if len(deg_inv) > 1 else 0,
            'gini': gini_coefficient(np.array(deg_inv)),
            'isolated': sum(1 for d in deg_inv if d == 0),
            'total': len(deg_inv)
        },
        'companies': {
            'degrees': deg_comp,
            'mean': np.mean(deg_comp),
            'median': np.median(deg_comp),
            'std': np.std(deg_comp),
            'max': np.max(deg_comp),
            'min': np.min(deg_comp),
            'skewness': skew(deg_comp) if len(deg_comp) > 1 else 0,
            'kurtosis': kurtosis(deg_comp) if len(deg_comp) > 1 else 0,
            'gini': gini_coefficient(np.array(deg_comp)),
            'isolated': sum(1 for d in deg_comp if d == 0),
            'total': len(deg_comp)
        }
    }

def calculate_bipartite_density(G, investors: List, companies: List) -> Dict:
    """Calcule les métriques de densité et sparsité"""
    m = G.number_of_edges()
    max_possible = len(investors) * len(companies)
    density = m / max_possible if max_possible > 0 else 0
    
    return {
        'edges': m,
        'max_possible_edges': max_possible,
        'density': density,
        'sparsity_percentage': (1 - density) * 100,
        'edges_per_investor': m / len(investors) if investors else 0,
        'edges_per_company': m / len(companies) if companies else 0
    }

def calculate_imbalance_metrics(G, investors: List, companies: List, deg_stats: Dict) -> Dict:
    """Calcule les métriques de déséquilibre"""
    n_inv = len(investors)
    n_comp = len(companies)
    
    # Ratio structurel
    ratio_ie = n_inv / n_comp if n_comp > 0 else float('inf')
    
    # Ratio de degré moyen
    avg_deg_inv = deg_stats['investors']['mean']
    avg_deg_comp = deg_stats['companies']['mean']
    deg_ratio = avg_deg_inv / avg_deg_comp if avg_deg_comp > 0 else float('inf')
    
    # Taux de déséquilibre pour prédiction (edge prediction imbalance)
    # Pour chaque investisseur, ratio liens existants / liens possibles
    imbalance_scores = []
    for investor in investors:
        existing_edges = G.degree(investor)
        possible_edges = len(companies)
        if possible_edges > 0:
            imbalance_scores.append(existing_edges / possible_edges)
    
    avg_positive_rate = np.mean(imbalance_scores) if imbalance_scores else 0
    neg_pos_ratio = (1 - avg_positive_rate) / avg_positive_rate if avg_positive_rate > 0 else float('inf')
    
    return {
        'ratio_investors_companies': ratio_ie,
        'ratio_avg_degree': deg_ratio,
        'avg_positive_rate': avg_positive_rate,
        'negative_positive_ratio': neg_pos_ratio,
        'size_imbalance': abs(n_inv - n_comp) / (n_inv + n_comp) if (n_inv + n_comp) > 0 else 0
    }

def calculate_connectivity_metrics(G, investors: List, companies: List) -> Dict:
    """Calcule les métriques de connectivité"""
    # Composantes connexes
    components = list(nx.connected_components(G))
    component_sizes = [len(c) for c in components]
    
    if component_sizes:
        largest_component = max(components, key=len)
        largest_size = len(largest_component)
        largest_percentage = largest_size / G.number_of_nodes() * 100
    else:
        largest_size = 0
        largest_percentage = 0
    
    # Assortativité sur les projections
    try:
        # Projection investisseurs
        G_inv = nx.projected_graph(G, investors)
        assortativity_inv = nx.degree_assortativity_coefficient(G_inv) if G_inv.number_of_edges() > 0 else 0
        
        # Projection entreprises
        G_comp = nx.projected_graph(G, companies)
        assortativity_comp = nx.degree_assortativity_coefficient(G_comp) if G_comp.number_of_edges() > 0 else 0
    except:
        assortativity_inv = 0
        assortativity_comp = 0
    
    return {
        'num_components': len(components),
        'largest_component_size': largest_size,
        'largest_component_percentage': largest_percentage,
        'component_sizes': component_sizes,
        'assortativity_investors': assortativity_inv,
        'assortativity_companies': assortativity_comp,
        'is_connected': nx.is_connected(G),
        'isolated_nodes': len(list(nx.isolates(G)))
    }

def calculate_temporal_metrics(events: List = None) -> Dict:
    """Calcule les métriques temporelles si des données temporelles sont disponibles"""
    if events is None or len(events) == 0:
        return {
            'has_temporal_data': False,
            'temporal_coverage': 0,
            'event_rate': 0
        }
    
    # Si tu as des données temporelles, les analyser ici
    timestamps = [e['timestamp'] for e in events]
    time_span = max(timestamps) - min(timestamps)
    
    return {
        'has_temporal_data': True,
        'num_events': len(events),
        'time_span_days': time_span / (24 * 3600) if time_span > 0 else 0,
        'events_per_day': len(events) / (time_span / (24 * 3600)) if time_span > 0 else 0,
        'min_timestamp': min(timestamps),
        'max_timestamp': max(timestamps)
    }

def calculate_power_law_fit(degrees: List) -> Dict:
    """Teste si la distribution suit une loi de puissance"""
    from scipy.optimize import curve_fit
    
    if len(degrees) < 10:
        return {'is_power_law': False, 'alpha': 0, 'xmin': 0}
    
    # Histogramme
    hist, bin_edges = np.histogram(degrees, bins='auto')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Filtrer les zéros
    mask = (hist > 0) & (bin_centers > 0)
    x = bin_centers[mask]
    y = hist[mask]
    
    if len(x) < 3:
        return {'is_power_law': False, 'alpha': 0, 'xmin': 0}
    
    # Ajustement puissance
    try:
        def power_law(x, alpha, C):
            return C * np.power(x, -alpha)
        
        # Log-log pour régression linéaire
        log_x = np.log(x)
        log_y = np.log(y)
        
        # Régression linéaire
        coeffs = np.polyfit(log_x, log_y, 1)
        alpha = -coeffs[0]  # Exposant de la loi de puissance
        
        # Calcul du R²
        y_pred = np.polyval(coeffs, log_x)
        ss_res = np.sum((log_y - y_pred) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'is_power_law': r_squared > 0.8,  # Seuil arbitraire
            'alpha': alpha,
            'r_squared': r_squared,
            'xmin': np.min(x),
            'xmax': np.max(x)
        }
    except:
        return {'is_power_law': False, 'alpha': 0, 'r_squared': 0}

# ===================================================================
# DIAGNOSTIC COMPLET
# ===================================================================

def comprehensive_graph_diagnosis(G, investors: List, companies: List) -> Dict:
    """
    Diagnostic complet du graphe avec évaluation des problèmes potentiels
    et recommandations pour le TGN.
    """
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLET DU GRAPHE POUR TGN")
    print("="*80)
    
    # 1. Calculer toutes les métriques
    print("\n📊 CALCUL DES MÉTRIQUES...")
    deg_stats = calculate_degree_statistics(G, investors, companies)
    density_stats = calculate_bipartite_density(G, investors, companies)
    imbalance_stats = calculate_imbalance_metrics(G, investors, companies, deg_stats)
    connectivity_stats = calculate_connectivity_metrics(G, investors, companies)
    
    # 2. Détecter les problèmes
    problems = []
    warnings = []
    recommendations = []
    
    print("\n🔍 DÉTECTION DES PROBLÈMES...")
    
    # A. Vérifier la sparsité
    if density_stats['density'] < 0.0001:
        problems.append("⚠️  HYPER SPARSE: densité < 0.01% - risque de sur-apprentissage élevé")
        recommendations.append("• Utiliser negative sampling intelligent avec hard negatives")
        recommendations.append("• Ajouter des features de graphe globales (PageRank, centralité)")
        recommendations.append("• Considérer data augmentation via métapaths")
    elif density_stats['density'] < 0.001:
        warnings.append("⚠️  Très sparse: densité < 0.1% - nécessite techniques spéciales")
        recommendations.append("• Augmenter le batch size pour mieux explorer l'espace")
        recommendations.append("• Utiliser des embeddings pré-entraînés si disponibles")
    
    # B. Vérifier l'inégalité des degrés (Gini)
    if deg_stats['investors']['gini'] > 0.8 or deg_stats['companies']['gini'] > 0.8:
        problems.append("⚠️  INÉGALITÉ EXTRÊME: Gini > 0.8 - quelques hubs dominent le réseau")
        recommendations.append("• Pondération inverse des fréquences dans la loss")
        recommendations.append("• Downsampling des hubs ou oversampling des nœuds périphériques")
        recommendations.append("• Utiliser des techniques robustes aux outliers")
    elif deg_stats['investors']['gini'] > 0.6 or deg_stats['companies']['gini'] > 0.6:
        warnings.append("⚠️  Inégalité forte: Gini > 0.6 - réseau très hétérogène")
        recommendations.append("• Normaliser les degrés dans les features")
    
    # C. Vérifier le déséquilibre pour edge prediction
    if imbalance_stats['avg_positive_rate'] < 0.01:
        problems.append(f"⚠️  DÉSÉQUILIBRE EXTRÊME: seulement {imbalance_stats['avg_positive_rate']*100:.2f}% de liens positifs")
        recommendations.append("• Utiliser Focal Loss ou autre loss adaptative")
        recommendations.append("• Oversampling agressif des positifs")
        recommendations.append("• Génération synthétique de positifs (SMOTE-like)")
    elif imbalance_stats['avg_positive_rate'] < 0.05:
        warnings.append(f"⚠️  Déséquilibre important: {imbalance_stats['avg_positive_rate']*100:.2f}% de positifs")
        recommendations.append("• Balanced batch sampling")
        recommendations.append("• Poids de classe dans la loss")
    
    # D. Vérifier les nœuds isolés
    isolated_total = deg_stats['investors']['isolated'] + deg_stats['companies']['isolated']
    total_nodes = len(investors) + len(companies)
    isolated_percentage = isolated_total / total_nodes * 100
    
    if isolated_percentage > 20:
        problems.append(f"⚠️  COLD START: {isolated_percentage:.1f}% de nœuds isolés")
        recommendations.append("• Ajouter des features externes (secteur, localisation)")
        recommendations.append("• Transfer learning depuis nœuds similaires")
        recommendations.append("• Modèle à deux étages: pré-entraînement sur sous-graphe connecté")
    elif isolated_percentage > 10:
        warnings.append(f"⚠️  Nombre significatif de nœuds isolés: {isolated_percentage:.1f}%")
    
    # E. Vérifier la connectivité
    if not connectivity_stats['is_connected'] and connectivity_stats['largest_component_percentage'] < 80:
        warnings.append(f"⚠️  Graphe fragmenté: plus grande composante = {connectivity_stats['largest_component_percentage']:.1f}%")
        recommendations.append("• Analyser chaque composante séparément si elles ont des dynamiques différentes")
        recommendations.append("• Se concentrer sur la plus grande composante pour l'entraînement")
    
    # F. Vérifier l'assortativité
    if abs(connectivity_stats['assortativity_investors']) > 0.4:
        warnings.append(f"⚠️  Assortativité forte chez les investisseurs: {connectivity_stats['assortativité_investors']:.2f}")
        recommendations.append("• Intégrer l'assortativité comme feature contextuelle")
    
    # 3. Calculer un score de santé
    print("\n💯 CALCUL DU SCORE DE SANTÉ...")
    health_score = 100
    
    # Pénalités
    if density_stats['density'] < 0.0001: health_score -= 40
    elif density_stats['density'] < 0.001: health_score -= 20
    elif density_stats['density'] < 0.01: health_score -= 10
    
    if deg_stats['investors']['gini'] > 0.8 or deg_stats['companies']['gini'] > 0.8: health_score -= 30
    elif deg_stats['investors']['gini'] > 0.6 or deg_stats['companies']['gini'] > 0.6: health_score -= 15
    
    if imbalance_stats['avg_positive_rate'] < 0.01: health_score -= 25
    elif imbalance_stats['avg_positive_rate'] < 0.05: health_score -= 10
    
    if isolated_percentage > 20: health_score -= 20
    elif isolated_percentage > 10: health_score -= 10
    
    if not connectivity_stats['is_connected'] and connectivity_stats['largest_component_percentage'] < 50:
        health_score -= 15
    
    health_score = max(0, health_score)
    
    # 4. Afficher le rapport
    print("\n" + "="*80)
    print("RAPPORT DE DIAGNOSTIC")
    print("="*80)
    
    print(f"\n📈 MÉTRIQUES CLÉS:")
    print(f"  • Investisseurs: {len(investors)}")
    print(f"  • Entreprises: {len(companies)}")
    print(f"  • Arêtes: {density_stats['edges']}")
    print(f"  • Densité: {density_stats['density']:.6f} ({density_stats['sparsity_percentage']:.1f}% de sparsité)")
    print(f"  • Ratio I/E: {imbalance_stats['ratio_investors_companies']:.2f}")
    print(f"  • Taux de positifs: {imbalance_stats['avg_positive_rate']*100:.4f}%")
    print(f"  • Ratio Nég/Pos: {imbalance_stats['negative_positive_ratio']:.1f}:1")
    print(f"  • Gini investisseurs: {deg_stats['investors']['gini']:.3f}")
    print(f"  • Gini entreprises: {deg_stats['companies']['gini']:.3f}")
    print(f"  • Nœuds isolés: {isolated_total} ({isolated_percentage:.1f}%)")
    print(f"  • Plus grande composante: {connectivity_stats['largest_component_percentage']:.1f}%")
    print(f"  • Assortativité investisseurs: {connectivity_stats['assortativity_investors']:.3f}")
    print(f"  • Assortativité entreprises: {connectivity_stats['assortativity_companies']:.3f}")
    
    print(f"\n📊 DISTRIBUTION DES DEGRÉS:")
    print(f"  Investisseurs: μ={deg_stats['investors']['mean']:.2f}, σ={deg_stats['investors']['std']:.2f}, "
          f"skew={deg_stats['investors']['skewness']:.2f}, kurt={deg_stats['investors']['kurtosis']:.2f}")
    print(f"  Entreprises: μ={deg_stats['companies']['mean']:.2f}, σ={deg_stats['companies']['std']:.2f}, "
          f"skew={deg_stats['companies']['skewness']:.2f}, kurt={deg_stats['companies']['kurtosis']:.2f}")
    
    # Analyser la distribution de puissance
    power_law_inv = calculate_power_law_fit(deg_stats['investors']['degrees'])
    power_law_comp = calculate_power_law_fit(deg_stats['companies']['degrees'])
    
    if power_law_inv['is_power_law']:
        print(f"  ✓ Distribution investisseurs suit une loi de puissance (α={power_law_inv['alpha']:.2f}, R²={power_law_inv['r_squared']:.2f})")
    if power_law_comp['is_power_law']:
        print(f"  ✓ Distribution entreprises suit une loi de puissance (α={power_law_comp['alpha']:.2f}, R²={power_law_comp['r_squared']:.2f})")
    
    print(f"\n🎯 SCORE DE SANTÉ: {health_score:.0f}/100")
    if health_score >= 80:
        print("  ✅ EXCELLENT - TGN devrait bien performer")
    elif health_score >= 60:
        print("  ⚠️  BON - Quelques ajustements nécessaires")
    elif health_score >= 40:
        print("  ⚠️  MOYEN - Techniques spéciales requises")
    else:
        print("  ❌ DIFFICILE - Repenser l'approche ou enrichir les données")
    
    if problems:
        print(f"\n❌ PROBLÈMES CRITIQUES ({len(problems)}):")
        for i, problem in enumerate(problems, 1):
            print(f"  {i}. {problem}")
    
    if warnings:
        print(f"\n⚠️  AVERTISSEMENTS ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    
    if recommendations:
        print(f"\n💡 RECOMMANDATIONS ({len(set(recommendations))}):")
        for i, rec in enumerate(sorted(set(recommendations)), 1):
            print(f"  {i}. {rec}")
    
    print("\n" + "="*80)
    
    # 5. Retourner tous les résultats
    return {
        'health_score': health_score,
        'problems': problems,
        'warnings': warnings,
        'recommendations': list(set(recommendations)),
        'metrics': {
            'degree_statistics': deg_stats,
            'density_statistics': density_stats,
            'imbalance_statistics': imbalance_stats,
            'connectivity_statistics': connectivity_stats,
            'power_law_analysis': {
                'investors': power_law_inv,
                'companies': power_law_comp
            }
        }
    }

# ===================================================================
# VISUALISATIONS DES MÉTRIQUES
# ===================================================================

def visualize_metrics_dashboard(diagnosis_results: Dict, save_path: str = None):
    """Crée un dashboard visuel des métriques de diagnostic"""
    metrics = diagnosis_results['metrics']
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Score de santé (gauge chart simplifié)
    ax1 = plt.subplot(3, 3, 1)
    health_score = diagnosis_results['health_score']
    
    # Créer un gauge chart simple
    colors = ['#FF6B6B', '#FFD166', '#06D6A0', '#118AB2']
    if health_score < 40:
        color = colors[0]
    elif health_score < 60:
        color = colors[1]
    elif health_score < 80:
        color = colors[2]
    else:
        color = colors[3]
    
    ax1.pie([health_score, 100-health_score], colors=[color, '#f0f0f0'], startangle=90)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    ax1.add_artist(centre_circle)
    ax1.text(0, 0, f'{health_score:.0f}', ha='center', va='center', fontsize=24, fontweight='bold')
    ax1.set_title('Score de Santé du Graphe', fontsize=14, fontweight='bold', pad=20)
    ax1.annotate('Difficile\n<40', xy=(-0.5, -0.8), ha='center', fontsize=9)
    ax1.annotate('Moyen\n40-60', xy=(-0.2, -1.0), ha='center', fontsize=9)
    ax1.annotate('Bon\n60-80', xy=(0.2, -1.0), ha='center', fontsize=9)
    ax1.annotate('Excellent\n≥80', xy=(0.5, -0.8), ha='center', fontsize=9)
    
    # 2. Distribution des degrés (log-log)
    ax2 = plt.subplot(3, 3, 2)
    deg_inv = metrics['degree_statistics']['investors']['degrees']
    deg_comp = metrics['degree_statistics']['companies']['degrees']
    
    # Histogramme log-log
    hist_inv, bins_inv = np.histogram(deg_inv, bins=50)
    bin_centers_inv = (bins_inv[:-1] + bins_inv[1:]) / 2
    hist_comp, bins_comp = np.histogram(deg_comp, bins=50)
    bin_centers_comp = (bins_comp[:-1] + bins_comp[1:]) / 2
    
    ax2.loglog(bin_centers_inv[hist_inv > 0], hist_inv[hist_inv > 0], 'ro-', 
               alpha=0.7, label='Investisseurs', markersize=4)
    ax2.loglog(bin_centers_comp[hist_comp > 0], hist_comp[hist_comp > 0], 'bo-', 
               alpha=0.7, label='Entreprises', markersize=4)
    
    ax2.set_xlabel('Degré (log)')
    ax2.set_ylabel('Fréquence (log)')
    ax2.set_title('Distribution des degrés (log-log)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Coefficients de Gini
    ax3 = plt.subplot(3, 3, 3)
    gini_inv = metrics['degree_statistics']['investors']['gini']
    gini_comp = metrics['degree_statistics']['companies']['gini']
    
    bars = ax3.bar(['Investisseurs', 'Entreprises'], [gini_inv, gini_comp], 
                   color=['#FF6B6B', '#118AB2'], alpha=0.7)
    ax3.set_ylabel('Coefficient de Gini')
    ax3.set_title('Inégalité des connexions', fontsize=12)
    ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Extrême (>0.8)')
    ax3.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Forte (>0.6)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs sur les barres
    for bar, val in zip(bars, [gini_inv, gini_comp]):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Déséquilibre edge prediction
    ax4 = plt.subplot(3, 3, 4)
    positive_rate = metrics['imbalance_statistics']['avg_positive_rate'] * 100
    negative_rate = 100 - positive_rate
    
    wedges, texts, autotexts = ax4.pie([positive_rate, negative_rate], 
                                       labels=['Positifs', 'Négatifs'],
                                       colors=['#06D6A0', '#FFD166'],
                                       autopct='%1.2f%%', startangle=90)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax4.set_title(f'Déséquilibre pour prédiction\n({metrics["imbalance_statistics"]["negative_positive_ratio"]:.1f}:1 Nég/Pos)', 
                  fontsize=12)
    
    # 5. Matrice des problèmes
    ax5 = plt.subplot(3, 3, 5)
    
    # Créer une matrice de sévérité
    problems = diagnosis_results.get('problems', [])
    warnings_list = diagnosis_results.get('warnings', [])
    
    categories = ['Sparsité', 'Inégalité', 'Déséquilibre', 'Connectivité', 'Cold Start']
    severity = np.zeros(len(categories))
    
    # Mapper les problèmes aux catégories
    for problem in problems:
        if 'SPARSE' in problem.upper():
            severity[0] = 2  # Critique
        elif 'INÉGALITÉ' in problem.upper():
            severity[1] = 2
        elif 'DÉSÉQUILIBRE' in problem.upper():
            severity[2] = 2
        elif 'COLD START' in problem.upper():
            severity[4] = 2
    
    for warning in warnings_list:
        if 'sparse' in warning.lower():
            severity[0] = max(severity[0], 1)  # Avertissement
        elif 'inégalité' in warning.lower():
            severity[1] = max(severity[1], 1)
        elif 'déséquilibre' in warning.lower():
            severity[2] = max(severity[2], 1)
        elif 'fragmenté' in warning.lower():
            severity[3] = max(severity[3], 1)
        elif 'isolés' in warning.lower():
            severity[4] = max(severity[4], 1)
    
    # Heatmap
    im = ax5.imshow([severity], cmap='YlOrRd', aspect='auto')
    ax5.set_xticks(range(len(categories)))
    ax5.set_xticklabels(categories, rotation=45, ha='right')
    ax5.set_yticks([0])
    ax5.set_yticklabels(['Sévérité'])
    ax5.set_title('Problèmes détectés', fontsize=12)
    
    # Ajouter les valeurs
    for i in range(len(categories)):
        if severity[i] > 0:
            text = 'CRITIQUE' if severity[i] == 2 else 'Avert.'
            color = 'white' if severity[i] == 2 else 'black'
            ax5.text(i, 0, text, ha='center', va='center', color=color, fontweight='bold')
    
    # 6. Comparaison degrés investisseurs vs entreprises
    ax6 = plt.subplot(3, 3, 6)
    
    deg_inv_mean = metrics['degree_statistics']['investors']['mean']
    deg_comp_mean = metrics['degree_statistics']['companies']['mean']
    
    x_pos = np.arange(2)
    bars = ax6.bar(x_pos, [deg_inv_mean, deg_comp_mean], 
                   color=['#FF6B6B', '#118AB2'], alpha=0.7)
    
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(['Investisseurs', 'Entreprises'])
    ax6.set_ylabel('Degré moyen')
    ax6.set_title('Comparaison degrés moyens', fontsize=12)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs
    for bar, val in zip(bars, [deg_inv_mean, deg_comp_mean]):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 7. Assortativité
    ax7 = plt.subplot(3, 3, 7)
    assort_inv = metrics['connectivity_statistics']['assortativity_investors']
    assort_comp = metrics['connectivity_statistics']['assortativity_companies']
    
    ax7.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax7.axhspan(-0.3, 0.3, alpha=0.1, color='gray')
    
    ax7.scatter([0, 1], [assort_inv, assort_comp], 
                s=200, c=['#FF6B6B', '#118AB2'], alpha=0.7)
    
    ax7.set_xticks([0, 1])
    ax7.set_xticklabels(['Investisseurs', 'Entreprises'])
    ax7.set_ylabel('Coefficient d\'assortativité')
    ax7.set_title('Assortativité des projections', fontsize=12)
    ax7.grid(True, alpha=0.3)
    
    # Ajouter les valeurs
    for i, val in enumerate([assort_inv, assort_comp]):
        ax7.text(i, val + 0.02 * np.sign(val), f'{val:.3f}', 
                ha='center', va='bottom' if val >= 0 else 'top', fontsize=10)
    
    # 8. Densité et sparsité
    ax8 = plt.subplot(3, 3, 8)
    density = metrics['density_statistics']['density'] * 100  # En pourcentage
    
    ax8.bar(['Densité'], [density], color='#06D6A0', alpha=0.7)
    ax8.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Très sparse')
    ax8.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='Sparse')
    ax8.axhline(y=10.0, color='green', linestyle='--', alpha=0.5, label='Dense')
    
    ax8.set_ylabel('Densité (%)')
    ax8.set_title(f'Densité du graphe\n{metrics["density_statistics"]["edges"]} arêtes / {metrics["density_statistics"]["max_possible_edges"]} possibles', 
                  fontsize=12)
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Ajouter la valeur
    ax8.text(0, density + 0.1, f'{density:.4f}%', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 9. Légende et informations
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
    RÉSUMÉ DU DIAGNOSTIC
    
    • Score de santé: {health_score:.0f}/100
    • Problèmes critiques: {len(problems)}
    • Avertissements: {len(warnings_list)}
    
    MÉTRIQUES CLÉS:
    • Nœuds: {len(deg_inv) + len(deg_comp)}
    • Arêtes: {metrics['density_statistics']['edges']}
    • Composantes: {metrics['connectivity_statistics']['num_components']}
    • Isolés: {metrics['connectivity_statistics']['isolated_nodes']}
    
    RECOMMANDATIONS:
    """
    
    # Limiter le nombre de recommandations affichées
    recs = diagnosis_results.get('recommendations', [])
    display_recs = recs[:3]  # Afficher seulement les 3 premières
    
    for i, rec in enumerate(display_recs, 1):
        summary_text += f"\n  {i}. {rec}"
    
    if len(recs) > 3:
        summary_text += f"\n  ... et {len(recs) - 3} de plus"
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle('DASHBOARD DE DIAGNOSTIC DU GRAPHE POUR TGN', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard sauvegardé: {save_path}")
    
    plt.show()

# ===================================================================
# UTILITAIRE POUR EXTRACTION DES NŒUDS
# ===================================================================

def extract_nodes(G, bipartite_set: int) -> List:
    """Extract nodes from one of the bipartite sets"""
    nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == bipartite_set]
    return nodes

# ===================================================================
# FONCTION PRINCIPALE
# ===================================================================

def main_graph_analysis():
    """Fonction principale pour l'analyse du graphe"""
    print("="*80)
    print("ANALYSE DE CARACTÉRISATION DU GRAPHE BIPARTI")
    print("="*80)
    
    # Charger les données (adapté à ton code existant)
    prefix = "cybersecurity_" if FLAG_CYBERSECURITY else ""
    
    # Charger le graphe
    graph_path = f'{SAVE_DIR_NETWORKS}/bipartite_graph_{NUM_COMP}.gpickle'
    with open(graph_path, 'rb') as f:
        B = pickle.load(f)
    
    # Charger les dictionnaires
    with open(f'{SAVE_DIR_CLASSES}/dict_companies_{NUM_COMP}.pickle', 'rb') as f:
        dict_companies = pickle.load(f)
    
    with open(f'{SAVE_DIR_CLASSES}/dict_investors_{NUM_TECH}.pickle', 'rb') as f:
        dict_tech = pickle.load(f)
    
    print(f"✓ Données chargées:")
    print(f"  - Graphe: {B.number_of_nodes()} nœuds, {B.number_of_edges()} arêtes")
    print(f"  - Dictionnaires: {len(dict_companies)} companies, {len(dict_tech)} investors")
    
    # Extraire les nœuds des deux partitions
    companies = extract_nodes(B, 0)  # Companies
    investors = extract_nodes(B, 1)  # Investors
    
    print(f"\n📊 COMPOSITION DU GRAPHE:")
    print(f"  - Companies: {len(companies)}")
    print(f"  - Investors: {len(investors)}")
    
    # Vérifier que c'est bien bipartite
    if not is_bipartite(B):
        print("❌ ERREUR: Le graphe n'est pas bipartite!")
        return
    
    # Exécuter le diagnostic complet
    diagnosis_results = comprehensive_graph_diagnosis(B, investors, companies)
    
    # Créer et sauvegarder le dashboard
    Path(SAVE_DIR_ANALYSIS).mkdir(parents=True, exist_ok=True)
    dashboard_path = f'{SAVE_DIR_ANALYSIS}/graph_diagnosis_dashboard.png'
    visualize_metrics_dashboard(diagnosis_results, dashboard_path)
    
    # Sauvegarder les résultats détaillés
    results_path = f'{SAVE_DIR_ANALYSIS}/diagnosis_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(diagnosis_results, f)
    
    print(f"\n✅ ANALYSE TERMINÉE")
    print(f"   - Dashboard: {dashboard_path}")
    print(f"   - Résultats détaillés: {results_path}")
    
    # Retourner les résultats pour utilisation ultérieure
    return diagnosis_results, B, companies, investors

# ===================================================================
# EXÉCUTION
# ===================================================================

if __name__ == "__main__":
    # Exécuter l'analyse complète
    results = main_graph_analysis()
    
    # Si tu veux accéder aux résultats dans le notebook:
    if results:
        diagnosis_results, B, companies, investors = results
        
        print("\n" + "="*80)
        print("UTILISATION DES RÉSULTATS POUR TGN")
        print("="*80)
        
        # Exemple d'utilisation des résultats pour configurer TGN
        health_score = diagnosis_results['health_score']
        metrics = diagnosis_results['metrics']
        
        print(f"\n🎯 CONFIGURATION RECOMMANDÉE POUR TGN:")
        
        if health_score >= 80:
            print("  • Utiliser l'architecture TGN standard")
            print("  • Batch size: 512")
            print("  • Negative sampling: 10 négatifs par positif")
            print("  • Pas besoin de techniques spéciales")
        elif health_score >= 60:
            print("  • Batch size: 1024 (plus grand pour explorer l'espace)")
            print("  • Negative sampling: 20 négatifs par positif")
            print("  • Ajouter des features de graphe globales")
        elif health_score >= 40:
            print("  • Batch size: 2048")
            print("  • Negative sampling: hard negative mining")
            print("  • Utiliser Focal Loss ou balanced loss")
            print("  • Ajouter régularisation forte")
        else:
            print("  • Repenser l'approche - le graphe est trop déséquilibré")
            print("  • Considérer data augmentation")
            print("  • Entraîner sur sous-ensembles")
            print("  • Utiliser transfer learning")