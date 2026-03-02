"""
models.py
=========
Implements all four classification models from Table IV:
  1. SVM (Support Vector Machine with RBF kernel)
  2. LSTM (Long Short-Term Memory neural network)
  3. Vanilla RF (Random Forest with default hyperparameters)
  4. GA-RF (Genetic Algorithm-optimized Random Forest)

Also includes K-Means workload profiling from the paper.
"""

import time
import warnings
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)

warnings.filterwarnings('ignore')


# ============================================================================
# 1. K-MEANS WORKLOAD PROFILER (Tier 2, Component 1)
# ============================================================================

def kmeans_profiler(X_train, X_test, k=5, random_state=42):
    """
    Cluster workloads into k profiles using K-Means.
    Appends cluster labels as an additional feature.
    """
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    train_clusters = kmeans.fit_predict(X_train)
    test_clusters = kmeans.predict(X_test)
    
    X_train_aug = np.column_stack([X_train, train_clusters])
    X_test_aug = np.column_stack([X_test, test_clusters])
    
    return X_train_aug, X_test_aug, kmeans


# ============================================================================
# 2. SVM CLASSIFIER
# ============================================================================

def train_svm(X_train, y_train, X_test, y_test):
    """Train SVM with RBF kernel and return metrics."""
    print("\n" + "="*60)
    print("  Training SVM (RBF Kernel)")
    print("="*60)
    
    start = time.time()
    svm = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred = svm.predict(X_test)
    
    metrics = {
        'model': 'SVM',
        'accuracy': accuracy_score(y_test, y_pred) * 100,
        'precision': precision_score(y_test, y_pred, average='weighted') * 100,
        'recall': recall_score(y_test, y_pred, average='weighted') * 100,
        'f1': f1_score(y_test, y_pred, average='weighted') * 100,
        'train_time': train_time,
    }
    
    print(f"  Accuracy:  {metrics['accuracy']:.1f}%")
    print(f"  F1-Score:  {metrics['f1']:.1f}%")
    print(f"  Time:      {metrics['train_time']:.1f}s")
    
    return svm, metrics, y_pred


# ============================================================================
# 3. LSTM CLASSIFIER
# ============================================================================

def train_lstm(X_train, y_train, X_test, y_test, epochs=30, batch_size=64):
    """Train LSTM model and return metrics."""
    print("\n" + "="*60)
    print("  Training LSTM (Deep Learning)")
    print("="*60)
    
    # Import TensorFlow here to avoid slow import if not needed
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
    
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM as LSTMLayer, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping
    
    n_classes = len(np.unique(y_train))
    y_train_cat = to_categorical(y_train, n_classes)
    y_test_cat = to_categorical(y_test, n_classes)
    
    # Reshape for LSTM: (samples, timesteps=1, features)
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    model = Sequential([
        LSTMLayer(64, input_shape=(1, X_train.shape[1]), return_sequences=True),
        Dropout(0.3),
        LSTMLayer(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    start = time.time()
    model.fit(
        X_train_lstm, y_train_cat,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    train_time = time.time() - start
    
    y_pred_prob = model.predict(X_test_lstm, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    metrics = {
        'model': 'LSTM',
        'accuracy': accuracy_score(y_test, y_pred) * 100,
        'precision': precision_score(y_test, y_pred, average='weighted') * 100,
        'recall': recall_score(y_test, y_pred, average='weighted') * 100,
        'f1': f1_score(y_test, y_pred, average='weighted') * 100,
        'train_time': train_time,
    }
    
    print(f"  Accuracy:  {metrics['accuracy']:.1f}%")
    print(f"  F1-Score:  {metrics['f1']:.1f}%")
    print(f"  Time:      {metrics['train_time']:.1f}s")
    
    return model, metrics, y_pred


# ============================================================================
# 4. VANILLA RANDOM FOREST (Default Hyperparameters)
# ============================================================================

def train_vanilla_rf(X_train, y_train, X_test, y_test):
    """Train Random Forest with default hyperparameters."""
    print("\n" + "="*60)
    print("  Training Vanilla Random Forest (Default Params)")
    print("="*60)
    
    start = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        criterion='gini',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred = rf.predict(X_test)
    
    metrics = {
        'model': 'Vanilla RF',
        'accuracy': accuracy_score(y_test, y_pred) * 100,
        'precision': precision_score(y_test, y_pred, average='weighted') * 100,
        'recall': recall_score(y_test, y_pred, average='weighted') * 100,
        'f1': f1_score(y_test, y_pred, average='weighted') * 100,
        'train_time': train_time,
    }
    
    print(f"  Accuracy:  {metrics['accuracy']:.1f}%")
    print(f"  F1-Score:  {metrics['f1']:.1f}%")
    print(f"  Time:      {metrics['train_time']:.1f}s")
    
    return rf, metrics, y_pred


# ============================================================================
# 5. GA-RF: GENETIC ALGORITHM-OPTIMIZED RANDOM FOREST
# ============================================================================

class GeneticAlgorithmRF:
    """
    Genetic Algorithm for Random Forest hyperparameter optimization.
    
    Optimizes: n_estimators, max_depth, min_samples_split,
               min_samples_leaf, max_features (ratio)
    
    Algorithm (from paper Section III.C.3):
      1. Initialize population randomly
      2. For each generation:
         a. Evaluate fitness (F1-score) on validation set
         b. Tournament selection (top 50%)
         c. Single-point crossover (p_c = 0.8)
         d. Gaussian mutation (p_m = 0.1)
         e. Elitism (preserve top 2)
      3. Return best hyperparameter set
    """
    
    def __init__(self, population_size=20, generations=25, crossover_prob=0.8,
                 mutation_prob=0.1, elitism_count=2, random_state=42):
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism_count = elitism_count
        self.random_state = random_state
        self.best_params = None
        self.best_fitness = 0
        self.fitness_history = []
        
        # Hyperparameter search bounds
        self.bounds = {
            'n_estimators': (50, 300),      # Number of trees
            'max_depth':    (5, 40),         # Tree depth
            'min_samples_split': (2, 10),    # Min split samples
            'min_samples_leaf':  (1, 8),     # Min leaf samples
            'max_features': (0.3, 0.9),      # Feature fraction
        }
    
    def _random_individual(self, rng):
        """Create a random chromosome (set of hyperparameters)."""
        return {
            'n_estimators': rng.integers(self.bounds['n_estimators'][0],
                                         self.bounds['n_estimators'][1]),
            'max_depth': rng.integers(self.bounds['max_depth'][0],
                                      self.bounds['max_depth'][1]),
            'min_samples_split': rng.integers(self.bounds['min_samples_split'][0],
                                               self.bounds['min_samples_split'][1]),
            'min_samples_leaf': rng.integers(self.bounds['min_samples_leaf'][0],
                                              self.bounds['min_samples_leaf'][1]),
            'max_features': round(rng.uniform(self.bounds['max_features'][0],
                                               self.bounds['max_features'][1]), 2),
        }
    
    def _evaluate_fitness(self, params, X_train, y_train, X_val, y_val):
        """Train RF with given params and return F1-score as fitness."""
        try:
            rf = RandomForestClassifier(
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                min_samples_split=int(params['min_samples_split']),
                min_samples_leaf=int(params['min_samples_leaf']),
                max_features=float(params['max_features']),
                bootstrap=True,
                criterion='gini',
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            return f1_score(y_val, y_pred, average='weighted')
        except Exception:
            return 0.0
    
    def _tournament_select(self, population, fitnesses, rng, k=3):
        """Tournament selection: pick k random, return the fittest."""
        idx = rng.choice(len(population), size=k, replace=False)
        best_idx = idx[np.argmax([fitnesses[i] for i in idx])]
        return population[best_idx].copy()
    
    def _crossover(self, parent1, parent2, rng):
        """Single-point crossover between two parents."""
        if rng.random() < self.crossover_prob:
            keys = list(parent1.keys())
            point = rng.integers(1, len(keys))
            child1, child2 = parent1.copy(), parent2.copy()
            for key in keys[point:]:
                child1[key], child2[key] = child2[key], child1[key]
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def _mutate(self, individual, rng):
        """Gaussian mutation on a random parameter."""
        if rng.random() < self.mutation_prob:
            key = rng.choice(list(individual.keys()))
            lo, hi = self.bounds[key]
            if key == 'max_features':
                individual[key] = round(np.clip(
                    individual[key] + rng.normal(0, 0.05), lo, hi
                ), 2)
            else:
                delta = rng.normal(0, (hi - lo) * 0.1)
                individual[key] = int(np.clip(individual[key] + delta, lo, hi))
        return individual
    
    def optimize(self, X_train, y_train, X_val, y_val):
        """Run the full GA optimization loop."""
        rng = np.random.default_rng(self.random_state)
        
        # Step 1: Initialize population
        population = [self._random_individual(rng) for _ in range(self.population_size)]
        
        print(f"\n  GA-RF Optimization: {self.generations} generations, "
              f"population={self.population_size}")
        print("  " + "-" * 50)
        
        for gen in range(self.generations):
            # Step 2a: Evaluate fitness
            fitnesses = [
                self._evaluate_fitness(ind, X_train, y_train, X_val, y_val)
                for ind in population
            ]
            
            # Track best
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]
            
            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_params = population[gen_best_idx].copy()
            
            self.fitness_history.append(self.best_fitness)
            
            if (gen + 1) % 5 == 0 or gen == 0:
                print(f"  Gen {gen+1:3d}/{self.generations}: "
                      f"Best F1 = {self.best_fitness:.4f} | "
                      f"Avg F1 = {np.mean(fitnesses):.4f}")
            
            # Step 2e: Elitism — preserve top individuals
            elite_idx = np.argsort(fitnesses)[-self.elitism_count:]
            elites = [population[i].copy() for i in elite_idx]
            
            # Step 2b: Selection + Crossover + Mutation
            new_population = list(elites)
            
            while len(new_population) < self.population_size:
                p1 = self._tournament_select(population, fitnesses, rng)
                p2 = self._tournament_select(population, fitnesses, rng)
                c1, c2 = self._crossover(p1, p2, rng)
                c1 = self._mutate(c1, rng)
                c2 = self._mutate(c2, rng)
                new_population.extend([c1, c2])
            
            population = new_population[:self.population_size]
        
        print(f"\n  ✓ GA Optimization Complete!")
        print(f"  Best F1-Score: {self.best_fitness:.4f}")
        print(f"  Best Params:   {self.best_params}")
        
        return self.best_params


def train_ga_rf(X_train, y_train, X_test, y_test,
                pop_size=20, generations=25):
    """Train GA-optimized Random Forest (the proposed model)."""
    print("\n" + "="*60)
    print("  Training GA-RF (Proposed — Genetic Algorithm Optimized)")
    print("="*60)
    
    # Split training data for GA validation
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Run GA optimization
    ga = GeneticAlgorithmRF(
        population_size=pop_size,
        generations=generations,
        crossover_prob=0.8,
        mutation_prob=0.1,
        elitism_count=2,
        random_state=42
    )
    
    start = time.time()
    best_params = ga.optimize(X_tr, y_tr, X_val, y_val)
    
    # Train final model with best params on full training data
    print("\n  Training final model with optimized hyperparameters...")
    ga_rf = RandomForestClassifier(
        n_estimators=int(best_params['n_estimators']),
        max_depth=int(best_params['max_depth']),
        min_samples_split=int(best_params['min_samples_split']),
        min_samples_leaf=int(best_params['min_samples_leaf']),
        max_features=float(best_params['max_features']),
        bootstrap=True,
        criterion='gini',
        random_state=42,
        n_jobs=-1
    )
    ga_rf.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred = ga_rf.predict(X_test)
    
    metrics = {
        'model': 'GA-RF (Proposed)',
        'accuracy': accuracy_score(y_test, y_pred) * 100,
        'precision': precision_score(y_test, y_pred, average='weighted') * 100,
        'recall': recall_score(y_test, y_pred, average='weighted') * 100,
        'f1': f1_score(y_test, y_pred, average='weighted') * 100,
        'train_time': train_time,
    }
    
    print(f"\n  ★ FINAL RESULTS (GA-RF):")
    print(f"  Accuracy:  {metrics['accuracy']:.1f}%")
    print(f"  F1-Score:  {metrics['f1']:.1f}%")
    print(f"  Time:      {metrics['train_time']:.1f}s")
    
    return ga_rf, metrics, y_pred, ga.fitness_history


if __name__ == "__main__":
    from data_generator import generate_workload_data, prepare_data
    
    X, y, names = generate_workload_data(n_samples=10000)
    X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler = prepare_data(X, y)
    
    # Augment with K-Means clusters
    X_train_aug, X_test_aug, km = kmeans_profiler(X_train_s, X_test_s)
    
    # Test each model
    _, svm_m, _ = train_svm(X_train_aug, y_train, X_test_aug, y_test)
    _, rf_m, _ = train_vanilla_rf(X_train_aug, y_train, X_test_aug, y_test)
    _, garf_m, _, _ = train_ga_rf(X_train_aug, y_train, X_test_aug, y_test,
                                    pop_size=10, generations=10)
    
    print("\n\nAll models trained successfully!")
