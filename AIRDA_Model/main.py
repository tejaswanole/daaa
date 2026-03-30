"""
main.py — AIRDA Framework Entry Point
======================================
Launches the Streamlit UI or runs the original CLI pipeline.

Usage:
    streamlit run ui/app.py          (recommended — launches the UI)
    python main.py                   (also launches the UI)
    python main.py --cli             (runs original CLI pipeline)
"""

import sys
import os
import subprocess


def launch_streamlit():
    """Launch the Streamlit application."""
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui', 'app.py')
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("╔════════════════════════════════════════════════════════╗")
    print("║  AIRDA Framework — Launching Streamlit UI...          ║")
    print("║  URL: http://localhost:8501                           ║")
    print("║  Press Ctrl+C to stop.                                ║")
    print("╚════════════════════════════════════════════════════════╝")
    
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path,
         "--server.headless", "true"],
        cwd=project_dir
    )


def run_cli():
    """Run the original CLI pipeline (from the core main.py logic)."""
    import time
    import warnings
    import numpy as np
    
    warnings.filterwarnings('ignore')
    
    # Add project root to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from core.data_generator import generate_workload_data, prepare_data
    from core.models import kmeans_profiler, train_svm, train_lstm, train_vanilla_rf, train_ga_rf
    from core.allocation_simulator import run_all_allocation_simulations
    
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  AIRDA: AI-Enabled Resource Detection & Allocation Framework     ║")
    print("║  CLI Mode — Running full pipeline                                ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    total_start = time.time()
    
    # Step 1: Generate Data
    print("\n[1/7] Generating synthetic workload data...")
    X, y, feature_names = generate_workload_data(n_samples=20000)
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = prepare_data(X, y)
    print(f"  → {len(y)} samples, {len(feature_names)} features")
    
    # Step 2: K-Means Profiling
    print("\n[2/7] K-Means workload clustering (k=5)...")
    X_train_aug, X_test_aug, kmeans = kmeans_profiler(X_train_s, X_test_s, k=5)
    print(f"  → Augmented features: {X_train_aug.shape[1]}")
    
    # Step 3: Train Models
    print("\n[3/7] Training all models...")
    svm_model, svm_m, svm_pred = train_svm(X_train_aug, y_train, X_test_aug, y_test)
    lstm_model, lstm_m, lstm_pred = train_lstm(X_train_aug, y_train, X_test_aug, y_test)
    rf_model, rf_m, rf_pred = train_vanilla_rf(X_train_aug, y_train, X_test_aug, y_test)
    garf_model, garf_m, garf_pred, ga_fitness = train_ga_rf(X_train_aug, y_train, X_test_aug, y_test)
    
    all_metrics = [svm_m, lstm_m, rf_m, garf_m]
    
    # Step 4: Print Table IV
    print("\n[4/7] Classification Performance (Table IV):")
    for m in all_metrics:
        print(f"  {m['model']:<20} Acc={m['accuracy']:.1f}%  F1={m['f1']:.1f}%  Time={m['train_time']:.1f}s")
    
    # Step 5: Allocation Simulation
    print("\n[5/7] Running allocation simulation...")
    alloc_results = run_all_allocation_simulations(X_test, y_test, svm_pred, lstm_pred, rf_pred, garf_pred)
    
    # Step 6: Generate plots
    print("\n[6/7] Generating plots...")
    from pipeline.evaluator import generate_all_plots, get_table_vii
    cd_df = get_table_vii(garf_model, X, y, feature_names, scaler)
    paths = generate_all_plots(all_metrics, alloc_results, garf_model, feature_names,
                               ga_fitness, garf_pred=garf_pred, y_test=y_test, cross_domain_df=cd_df)
    print(f"  → {len(paths)} plots saved to outputs/")
    
    # Step 7: Done
    total_time = time.time() - total_start
    print(f"\n[7/7] ✅ Done! Total time: {total_time:.1f}s")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_cli()
    else:
        launch_streamlit()
