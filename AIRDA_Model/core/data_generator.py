"""
data_generator.py
=================
Generates synthetic cloud workload data simulating the 9-dimensional feature
vectors described in the AIRDA research paper.

Features:
  x_t = [cpu, mem, disk_read, disk_write, net_in, net_out, task_queue, delta_cpu, delta_mem]

Labels (Allocation Tiers):
  0 = Low, 1 = Medium, 2 = High, 3 = Critical
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_workload_data(n_samples=20000, random_state=42):
    """
    Generate synthetic workload data with 9 features and 4 allocation classes.
    
    The data is constructed so that:
    - Class 0 (Low):      Low CPU/Memory, minimal I/O
    - Class 1 (Medium):   Moderate CPU/Memory, some I/O
    - Class 2 (High):     High CPU/Memory, significant I/O
    - Class 3 (Critical): Very high CPU/Memory, heavy I/O, large task queues
    
    Returns:
        X (np.ndarray): Feature matrix (n_samples x 9)
        y (np.ndarray): Labels (n_samples,)
        feature_names (list): Names of the 9 features
    """
    np.random.seed(random_state)
    
    feature_names = [
        'cpu_util', 'mem_usage', 'disk_read', 'disk_write',
        'net_in', 'net_out', 'task_queue', 'delta_cpu', 'delta_mem'
    ]
    
    samples_per_class = n_samples // 4
    
    # --- Class 0: Low ---
    low = np.column_stack([
        np.random.uniform(5, 25, samples_per_class),       # cpu: 5-25%
        np.random.uniform(10, 30, samples_per_class),       # mem: 10-30%
        np.random.uniform(0, 5, samples_per_class),         # disk_read: 0-5 MB/s
        np.random.uniform(0, 3, samples_per_class),         # disk_write: 0-3 MB/s
        np.random.uniform(0, 10, samples_per_class),        # net_in: 0-10 MB/s
        np.random.uniform(0, 8, samples_per_class),         # net_out: 0-8 MB/s
        np.random.randint(0, 5, samples_per_class),         # task_queue: 0-5
        np.random.normal(0, 2, samples_per_class),          # delta_cpu: near 0
        np.random.normal(0, 1.5, samples_per_class),        # delta_mem: near 0
    ])
    
    # --- Class 1: Medium ---
    medium = np.column_stack([
        np.random.uniform(25, 55, samples_per_class),       # cpu: 25-55%
        np.random.uniform(30, 55, samples_per_class),       # mem: 30-55%
        np.random.uniform(5, 30, samples_per_class),        # disk_read
        np.random.uniform(3, 20, samples_per_class),        # disk_write
        np.random.uniform(10, 50, samples_per_class),       # net_in
        np.random.uniform(8, 40, samples_per_class),        # net_out
        np.random.randint(5, 20, samples_per_class),        # task_queue
        np.random.normal(2, 4, samples_per_class),          # delta_cpu: slight increase
        np.random.normal(1, 3, samples_per_class),          # delta_mem
    ])
    
    # --- Class 2: High ---
    high = np.column_stack([
        np.random.uniform(55, 80, samples_per_class),       # cpu: 55-80%
        np.random.uniform(55, 80, samples_per_class),       # mem: 55-80%
        np.random.uniform(30, 80, samples_per_class),       # disk_read
        np.random.uniform(20, 60, samples_per_class),       # disk_write
        np.random.uniform(50, 150, samples_per_class),      # net_in
        np.random.uniform(40, 120, samples_per_class),      # net_out
        np.random.randint(20, 50, samples_per_class),       # task_queue
        np.random.normal(5, 5, samples_per_class),          # delta_cpu: increasing
        np.random.normal(3, 4, samples_per_class),          # delta_mem
    ])
    
    # --- Class 3: Critical ---
    critical = np.column_stack([
        np.random.uniform(80, 100, samples_per_class),      # cpu: 80-100%
        np.random.uniform(80, 100, samples_per_class),      # mem: 80-100%
        np.random.uniform(80, 200, samples_per_class),      # disk_read
        np.random.uniform(60, 150, samples_per_class),      # disk_write
        np.random.uniform(150, 500, samples_per_class),     # net_in
        np.random.uniform(120, 400, samples_per_class),     # net_out
        np.random.randint(50, 100, samples_per_class),      # task_queue
        np.random.normal(10, 6, samples_per_class),         # delta_cpu: spiking
        np.random.normal(8, 5, samples_per_class),          # delta_mem
    ])
    
    X = np.vstack([low, medium, high, critical])
    y = np.concatenate([
        np.zeros(samples_per_class),
        np.ones(samples_per_class),
        np.full(samples_per_class, 2),
        np.full(samples_per_class, 3),
    ]).astype(int)
    
    # Add 5% noise/overlap to make it realistic (not perfectly separable)
    noise_idx = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    for idx in noise_idx:
        # Randomly swap some feature values with adjacent class
        swap_class = min(3, max(0, y[idx] + np.random.choice([-1, 1])))
        if swap_class != y[idx]:
            # Add partial noise but keep label — this makes it harder to classify
            blend = np.random.uniform(0.3, 0.7)
            other_sample = np.random.randint(
                int(swap_class * samples_per_class),
                int((swap_class + 1) * samples_per_class)
            )
            X[idx] = blend * X[idx] + (1 - blend) * X[other_sample]
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y, feature_names


def prepare_data(X, y, test_size=0.2, random_state=42):
    """Split and scale data for training/testing."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    X, y, names = generate_workload_data()
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Feature names: {names}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"\nSample feature vector:\n{X[0]}")
    print(f"Sample label: {y[0]} ({'Low' if y[0]==0 else 'Medium' if y[0]==1 else 'High' if y[0]==2 else 'Critical'})")
