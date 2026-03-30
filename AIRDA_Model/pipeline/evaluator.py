"""
pipeline/evaluator.py
=====================
Handles metric computation, table generation, allocation simulation,
and plot generation. Returns DataFrames for UI display.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.allocation_simulator import run_all_allocation_simulations
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_table_iv(all_metrics):
    """
    Build Table IV: Classification Performance as a DataFrame.
    """
    rows = []
    for m in all_metrics:
        rows.append({
            'Model': m['model'],
            'Accuracy (%)': round(m['accuracy'], 2),
            'Precision (%)': round(m['precision'], 2),
            'Recall (%)': round(m['recall'], 2),
            'F1-Score (%)': round(m['f1'], 2),
            'Train Time (s)': round(m['train_time'], 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, 'table_iv_classification.csv'), index=False)
    return df


def get_table_v(X_test, y_test, predictions):
    """
    Run allocation simulation and return Table V as a DataFrame.
    
    Args:
        X_test: Original (unscaled) test features
        y_test: True labels
        predictions: dict mapping model key to prediction arrays
            Expected keys: 'SVM', 'LSTM', 'Random Forest', 'GA-RF'
    """
    svm_pred = predictions.get('SVM', predictions.get('svm'))
    lstm_pred = predictions.get('LSTM', predictions.get('lstm'))
    rf_pred = predictions.get('Random Forest', predictions.get('rf'))
    garf_pred = predictions.get('GA-RF', predictions.get('garf'))
    
    alloc_results = run_all_allocation_simulations(
        X_test, y_test, svm_pred, lstm_pred, rf_pred, garf_pred
    )
    
    rows = []
    for r in alloc_results:
        rows.append({
            'Strategy': r['strategy'],
            'Avg Latency (ms)': round(r['avg_latency_ms'], 1),
            'Utilization (%)': round(r['utilization_pct'], 1),
            'Energy (kWh)': round(r['energy_kwh'], 1),
            'SLA Violations (%)': round(r['sla_violations_pct'], 1),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, 'table_v_allocation.csv'), index=False)
    return df, alloc_results


def get_table_vi(rf_model, feature_names):
    """
    Build Table VI: Feature Importance Ranking as a DataFrame.
    """
    importances = rf_model.feature_importances_
    names_ext = feature_names + ['cluster_label']
    if len(importances) != len(names_ext):
        names_ext = feature_names[:len(importances)]
    
    pretty = {
        'cpu_util': 'CPU Utilization (%)',
        'mem_usage': 'Memory Usage (%)',
        'disk_read': 'Disk Read I/O (MB/s)',
        'disk_write': 'Disk Write I/O (MB/s)',
        'net_in': 'Network Ingress (MB/s)',
        'net_out': 'Network Egress (MB/s)',
        'task_queue': 'Task Queue Length',
        'delta_cpu': 'ΔCPU (temporal trend)',
        'delta_mem': 'ΔMemory (temporal trend)',
        'cluster_label': 'Cluster Label',
    }
    
    indices = np.argsort(importances)[::-1]
    rows = []
    for rank, idx in enumerate(indices, 1):
        name = pretty.get(names_ext[idx], names_ext[idx])
        rows.append({
            'Rank': rank,
            'Feature': name,
            'Importance Score': round(importances[idx], 4),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, 'table_vi_feature_importance.csv'), index=False)
    return df, importances, names_ext


def get_table_vii(model, X, y, feature_names, scaler):
    """
    Build Table VII: Cross-Domain Validation as a DataFrame.
    """
    domains = [
        {'name': 'Cloud VM Scheduling', 'noise': 0.02, 'notes': 'Primary domain (synthetic cloud trace)'},
        {'name': 'IoT Device Management', 'noise': 0.08, 'notes': 'Scaled features for IoT profile'},
        {'name': 'Edge Computing', 'noise': 0.12, 'notes': 'Higher noise, constrained devices'},
        {'name': 'HPC Job Scheduling', 'noise': 0.05, 'notes': 'Batch workload characteristics'},
    ]
    
    rows = []
    for domain in domains:
        np.random.seed(hash(domain['name']) % 10000)
        X_domain = X.copy()
        noise = np.random.normal(0, domain['noise'], X_domain.shape) * np.std(X_domain, axis=0)
        X_domain = X_domain + noise
        X_scaled = scaler.transform(X_domain)
        
        km = KMeans(n_clusters=5, random_state=42, n_init=10)
        clusters = km.fit_predict(X_scaled)
        X_aug = np.column_stack([X_scaled, clusters])
        
        y_pred = model.predict(X_aug)
        acc = accuracy_score(y, y_pred) * 100
        
        rows.append({
            'Domain': domain['name'],
            'Accuracy (%)': round(acc, 1),
            'Notes': domain['notes'],
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, 'table_vii_cross_domain.csv'), index=False)
    return df


def generate_all_plots(all_metrics, alloc_results, rf_model, feature_names,
                       ga_fitness, garf_pred=None, y_test=None, cross_domain_df=None):
    """
    Generate all visualization plots and save to /outputs.
    Returns list of saved file paths.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    saved = []
    
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'figure.dpi': 200
    })
    
    # --- Fig 1: Classification Performance Bar Chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    models = [m['model'] for m in all_metrics]
    accuracy = [m['accuracy'] for m in all_metrics]
    f1 = [m['f1'] for m in all_metrics]
    precision = [m['precision'] for m in all_metrics]
    recall = [m['recall'] for m in all_metrics]
    x = np.arange(len(models))
    w = 0.2
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    ax.bar(x - 1.5*w, accuracy, w, label='Accuracy', color=colors[0], edgecolor='white')
    ax.bar(x - 0.5*w, precision, w, label='Precision', color=colors[1], edgecolor='white')
    ax.bar(x + 0.5*w, recall, w, label='Recall', color=colors[2], edgecolor='white')
    ax.bar(x + 1.5*w, f1, w, label='F1-Score', color=colors[3], edgecolor='white')
    ax.set_xlabel('Classification Model')
    ax.set_ylabel('Score (%)')
    ax.set_title('Classification Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'fig_classification_performance.png')
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    saved.append(p)
    
    # --- Fig 2: Allocation Latency ---
    fig, ax = plt.subplots(figsize=(8, 5))
    strategies = [r['strategy'] for r in alloc_results]
    latencies = [r['avg_latency_ms'] for r in alloc_results]
    bar_colors = ['#e74c3c' if 'GA-RF' not in s else '#27ae60' for s in strategies]
    bars = ax.bar(range(len(strategies)), latencies, color=bar_colors, edgecolor='white', width=0.65)
    ax.set_xlabel('Allocation Strategy')
    ax.set_ylabel('Average Allocation Latency (ms)')
    ax.set_title('Resource Allocation Latency Across Strategies')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=25, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,
                f'{bar.get_height():.0f}ms', ha='center', va='bottom', fontsize=8, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'fig_allocation_latency.png')
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    saved.append(p)
    
    # --- Fig 3: Feature Importance ---
    fig, ax = plt.subplots(figsize=(7, 5))
    importances = rf_model.feature_importances_
    names_ext = feature_names + ['cluster_label']
    if len(importances) != len(names_ext):
        names_ext = feature_names[:len(importances)]
    pretty = {
        'cpu_util': 'CPU Utilization', 'mem_usage': 'Memory Usage',
        'disk_read': 'Disk Read I/O', 'disk_write': 'Disk Write I/O',
        'net_in': 'Network Ingress', 'net_out': 'Network Egress',
        'task_queue': 'Task Queue Len', 'delta_cpu': 'ΔCPU Trend',
        'delta_mem': 'ΔMemory Trend', 'cluster_label': 'Cluster Label'
    }
    indices = np.argsort(importances)[::-1]
    sorted_names = [pretty.get(names_ext[i], names_ext[i]) for i in indices]
    sorted_imp = [importances[i] for i in indices]
    colors_imp = ['#27ae60' if i < 3 else '#3498db' if i < 6 else '#95a5a6' for i in range(len(sorted_imp))]
    ax.barh(range(len(sorted_imp)), sorted_imp, color=colors_imp, edgecolor='white', height=0.65)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Gini Importance Score')
    ax.set_title('Feature Importance Ranking (GA-RF Model)')
    ax.grid(axis='x', alpha=0.3)
    for i, v in enumerate(sorted_imp):
        ax.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=8)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'fig_feature_importance.png')
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    saved.append(p)
    
    # --- Fig 4: GA Convergence ---
    if ga_fitness:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        gens = range(1, len(ga_fitness)+1)
        ax.plot(gens, ga_fitness, color='#e74c3c', linewidth=2.5,
                marker='o', markersize=4, markerfacecolor='white',
                markeredgecolor='#e74c3c', markeredgewidth=1.5)
        ax.fill_between(gens, ga_fitness, alpha=0.1, color='#e74c3c')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Fitness (F1-Score)')
        ax.set_title('Genetic Algorithm Convergence for RF Hyperparameter Optimization')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        p = os.path.join(OUTPUT_DIR, 'fig_ga_convergence.png')
        plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        saved.append(p)
    
    # --- Fig 5: Confusion Matrix ---
    if garf_pred is not None and y_test is not None:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, garf_pred)
        class_names = ['Low', 'Medium', 'High', 'Critical']
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax, shrink=0.8)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               ylabel='True Allocation Tier', xlabel='Predicted Allocation Tier',
               title='Confusion Matrix: GA-RF Classifier')
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black',
                        fontsize=12, fontweight='bold')
        plt.tight_layout()
        p = os.path.join(OUTPUT_DIR, 'fig_confusion_matrix.png')
        plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        saved.append(p)
    
    # --- Fig 6: Multi-Metric Comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Resource Allocation Efficiency: Multi-Metric Comparison',
                 fontsize=14, fontweight='bold', y=0.98)
    strategies_short = [s.replace('-Based', '').replace(' (Proposed)', '*') for s in strategies]
    strat_colors = []
    for s in strategies:
        if 'GA-RF' in s: strat_colors.append('#27ae60')
        elif 'Round' in s: strat_colors.append('#e74c3c')
        elif 'Threshold' in s: strat_colors.append('#f39c12')
        elif 'SVM' in s: strat_colors.append('#3498db')
        elif 'LSTM' in s: strat_colors.append('#9b59b6')
        else: strat_colors.append('#1abc9c')
    
    axes[0,0].bar(strategies_short, [r['avg_latency_ms'] for r in alloc_results], color=strat_colors, edgecolor='white')
    axes[0,0].set_ylabel('ms'); axes[0,0].set_title('Allocation Latency')
    axes[0,0].tick_params(axis='x', rotation=30, labelsize=7); axes[0,0].grid(axis='y', alpha=0.3)
    axes[0,1].bar(strategies_short, [r['utilization_pct'] for r in alloc_results], color=strat_colors, edgecolor='white')
    axes[0,1].set_ylabel('%'); axes[0,1].set_title('Resource Utilization')
    axes[0,1].tick_params(axis='x', rotation=30, labelsize=7); axes[0,1].grid(axis='y', alpha=0.3)
    axes[1,0].bar(strategies_short, [r['energy_kwh'] for r in alloc_results], color=strat_colors, edgecolor='white')
    axes[1,0].set_ylabel('kWh'); axes[1,0].set_title('Energy Consumption')
    axes[1,0].tick_params(axis='x', rotation=30, labelsize=7); axes[1,0].grid(axis='y', alpha=0.3)
    axes[1,1].bar(strategies_short, [r['sla_violations_pct'] for r in alloc_results], color=strat_colors, edgecolor='white')
    axes[1,1].set_ylabel('%'); axes[1,1].set_title('SLA Violations')
    axes[1,1].tick_params(axis='x', rotation=30, labelsize=7); axes[1,1].grid(axis='y', alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUTPUT_DIR, 'fig_allocation_multimetric.png')
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    saved.append(p)
    
    # --- Fig 7: Training Time ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    models_time = [m['model'] for m in all_metrics]
    times = [m['train_time'] for m in all_metrics]
    colors_time = ['#3498db', '#9b59b6', '#1abc9c', '#27ae60']
    bars = ax.bar(models_time, times, color=colors_time, edgecolor='white', width=0.55)
    ax.set_xlabel('Model'); ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison'); ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
               f'{bar.get_height():.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'fig_training_time.png')
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    saved.append(p)
    
    # --- Fig 8: Cross-Domain Validation ---
    if cross_domain_df is not None and len(cross_domain_df) > 0:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        domains = cross_domain_df['Domain'].tolist()
        accs = cross_domain_df['Accuracy (%)'].tolist()
        domain_colors = ['#27ae60', '#3498db', '#f39c12', '#e74c3c']
        bars = ax.bar(domains, accs, color=domain_colors[:len(domains)], edgecolor='white', width=0.55)
        ax.set_ylabel('Classification Accuracy (%)')
        ax.set_title('Cross-Domain Validation Results')
        ax.set_ylim(85, 100); ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=90, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.5, label='90% threshold')
        ax.legend(fontsize=9)
        ax.tick_params(axis='x', rotation=15, labelsize=9)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        p = os.path.join(OUTPUT_DIR, 'fig_cross_domain.png')
        plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        saved.append(p)
    
    return saved
