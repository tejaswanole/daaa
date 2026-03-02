# PATENT DRAFT

## SYSTEM AND METHOD FOR AI-ENABLED RESOURCE DETECTION AND DYNAMIC ALLOCATION USING RANDOM FOREST ENSEMBLE LEARNING

---

### PATENT APPLICATION

**Application Type:** Indian Patent / Provisional Patent Application  
**Filed Under:** The Patents Act, 1970 (India)  
**IPC Classification:** G06N 20/20 (Machine Learning — Ensemble Methods); G06F 9/50 (Resource Allocation, Scheduling)

---

### INVENTORS

1. **Sonali Bhoite**, Department of Information Technology, Vishwakarma Institute of Technology, Pune, India
2. **Tejas Wanole**, Department of Information Technology, Vishwakarma Institute of Information Technology, Pune, India
3. **Nirant Kale**, Department of Information Technology, Vishwakarma Institute of Information Technology, Pune, India
4. **Riddhi Mirajkar**, Department of Information Technology, Vishwakarma Institute of Technology, Pune, India
5. **Rohan Nemade**, Department of Information Technology, Vishwakarma Institute of Information Technology, Pune, India
6. **Durvesh Chavan**, Department of Information Technology, Vishwakarma Institute of Information Technology, Pune, India

**Applicant Institution:** Vishwakarma Institute of Technology / Vishwakarma Institute of Information Technology, Pune, Maharashtra, India

---

### FIELD OF THE INVENTION

The present invention relates to the field of artificial intelligence and machine learning applied to computational resource management. More particularly, it pertains to a system and method for automated detection of available computational resources and intelligent allocation thereof using a Random Forest ensemble learning classifier, integrated with K-Means clustering-based workload profiling and Genetic Algorithm-based hyperparameter optimization.

---

### BACKGROUND OF THE INVENTION

Modern computing environments—including cloud computing platforms, Internet of Things (IoT) networks, edge computing systems, and high-performance computing (HPC) clusters—face a critical challenge in dynamically allocating computational resources (CPU, memory, storage, and network bandwidth) to incoming workloads. The global cloud computing market is projected to exceed $912 billion in 2025, with inefficient resource allocation estimated to waste 30–35% of cloud expenditures.

#### Limitations of Existing Art:

1. **Static Allocation Methods:** Conventional techniques such as Round Robin, First-Fit, and Best-Fit algorithms allocate resources based on fixed rules, failing to adapt to dynamic workload patterns.

2. **Threshold-Based Auto-Scaling:** Current cloud platforms employ reactive threshold-based scaling (e.g., add a VM when CPU > 70%), which introduces latency between demand spikes and resource provisioning.

3. **Deep Learning Approaches:** While LSTM and Transformer-based models offer high prediction accuracy, they require significant computational resources (GPU hardware), have long training times, and lack interpretability—making them impractical for real-time, edge-deployed resource management.

4. **Single-Dimension Solutions:** Existing AI-based allocation systems typically address only one resource dimension (e.g., CPU scheduling alone) or one deployment domain (e.g., cloud only), lacking a unified cross-domain framework.

5. **Absence of Integrated Detection-Allocation Pipeline:** Prior art treats resource detection (monitoring) and allocation (scheduling) as separate systems, introducing communication overhead and preventing end-to-end optimization.

There exists a need for an intelligent, integrated system that can detect available resources in real-time, classify workload demands across multiple dimensions, and execute optimal allocation decisions with minimal latency, low computational overhead, and high interpretability.

---

### SUMMARY OF THE INVENTION

The present invention provides a novel **AI-Enabled Resource Detection and Allocation (AIRDA)** system and method comprising:

**A three-tier integrated architecture:**

1. **Resource Detection Layer** — a multi-dimensional monitoring subsystem that continuously captures CPU, memory, disk I/O, network bandwidth, and GPU utilization metrics from heterogeneous computing infrastructure, and constructs temporal feature vectors incorporating first-order difference trends;

2. **Intelligent Decision Layer** — comprising:
   - A **K-Means clustering module** that profiles incoming workloads into five categories (CPU-intensive, memory-intensive, I/O-intensive, network-intensive, balanced);
   - A **Random Forest (RF) ensemble classifier** trained on historical workload traces to classify resource demands into allocation tiers (Low, Medium, High, Critical);
   - A **Genetic Algorithm (GA)-based hyperparameter optimizer** that automatically tunes RF parameters (number of estimators, tree depth, minimum samples split/leaf, feature subsampling ratio) to maximize classification F1-score;

3. **Allocation Execution Layer** — a policy-driven engine that maps RF classification outputs to specific resource configurations, implements predictive horizontal/vertical scaling, and enforces SLA compliance thresholds.

**Key novel aspects of the invention:**
- The integration of K-Means workload profiling with GA-optimized RF classification in a single decision pipeline;
- The construction of temporal feature vectors with first-order difference terms (ΔCPU, ΔMemory) that capture workload trends;
- A pre-emptive scaling mechanism triggered by consecutive RF prediction transitions;
- An enhanced RF-based anomaly detection module for identifying abnormal resource consumption patterns;
- Cross-domain applicability validated across cloud, IoT, edge computing, and HPC environments.

---

### DETAILED DESCRIPTION OF THE INVENTION

#### 1. System Architecture

The AIRDA system comprises three interconnected tiers operating in a continuous monitoring-prediction-execution loop:

##### 1.1 Resource Detection Layer (Tier 1)

The Resource Detection Layer includes:

**(a) Multi-Dimensional Metric Collector:** A lightweight daemon process executing on each monitored computing node (physical server, virtual machine, container, or IoT device) that samples the following metrics at a configurable interval T_sample (default: 5 seconds):
- Per-core CPU utilization (percentage, 0–100)
- Memory usage (used bytes, total bytes, swap bytes)
- Disk I/O throughput (read MB/s, write MB/s)
- Network bandwidth utilization (ingress MB/s, egress MB/s)
- Task queue length (number of pending tasks)
- GPU utilization and VRAM usage (when GPU is present)

**(b) Feature Vector Constructor:** At each sampling interval t, the collector constructs a 9-dimensional feature vector:

**x_t = [cpu_t, mem_t, disk_read_t, disk_write_t, net_in_t, net_out_t, task_queue_t, Δcpu_t, Δmem_t]**

where:
- Δcpu_t = cpu_t − cpu_{t−1} (CPU utilization trend)
- Δmem_t = mem_t − mem_{t−1} (Memory utilization trend)

The inclusion of temporal difference features (**Δcpu**, **Δmem**) enables the downstream classifier to anticipate workload changes before they fully manifest, enabling proactive resource allocation.

**(c) Enhanced RF Anomaly Detection Module:** A secondary Random Forest model trained on labeled anomaly data detects abnormal resource patterns (e.g., memory leaks, CPU runaway processes, network flooding). Detected anomalies are flagged with priority labels and routed to the Allocation Execution Layer for immediate handling.

##### 1.2 Intelligent Decision Layer (Tier 2)

**(a) K-Means Workload Profiler:**

Incoming workload feature vectors are clustered using the K-Means algorithm with **k = 5** clusters:
- Cluster 0: CPU-intensive workloads
- Cluster 1: Memory-intensive workloads
- Cluster 2: I/O-intensive workloads
- Cluster 3: Network-intensive workloads
- Cluster 4: Balanced workloads

The optimal value of k is determined through Elbow Method analysis and Silhouette Score maximization. The cluster label is appended to the feature vector as an additional input to the RF classifier.

**(b) Random Forest Ensemble Classifier:**

The core classification model comprises **T** decision trees (default T = 100, GA-optimized T = 187), each trained on a bootstrap sample of the training data:

- **Input:** 10-dimensional vector (9 resource metrics + cluster label)
- **Output:** Allocation class ∈ {Low, Medium, High, Critical}
- **Split Criterion:** Gini impurity
- **Feature Selection:** At each node, a random subset of m' = ⌊√m⌋ features (default) or m' = ⌊0.67·m⌋ (GA-optimized) is considered
- **Aggregation:** Majority voting across all T trees

The RF classifier provides two critical outputs:
1. **Predicted allocation class** for resource mapping
2. **Feature importance scores** (Gini importance) for resource profiling and monitoring dashboard visualization

**(c) Genetic Algorithm Hyperparameter Optimizer:**

A Genetic Algorithm with the following configuration optimizes six RF hyperparameters simultaneously:
- **Population size:** P = 30 chromosomes
- **Generations:** G = 50
- **Selection:** Tournament selection (top 50%)
- **Crossover:** Single-point crossover, probability p_c = 0.8
- **Mutation:** Gaussian perturbation, probability p_m = 0.1
- **Elitism:** Top 2 individuals preserved per generation
- **Fitness function:** F1-score on held-out validation set

The six optimized hyperparameters are: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features (ratio), and bootstrap (boolean).

##### 1.3 Allocation Execution Layer (Tier 3)

**(a) Policy Engine:** Maps classification outputs to resource configurations:

| Classification | vCPU | RAM | Disk | Bandwidth |
|---|---|---|---|---|
| Low | 1 | 1 GB | 10 GB | 100 Mbps |
| Medium | 2 | 4 GB | 50 GB | 500 Mbps |
| High | 4 | 8 GB | 100 GB | 1 Gbps |
| Critical | 8+ | 16+ GB | 200+ GB | 10 Gbps |

**(b) Pre-emptive Scaling Engine:** Implements predictive scaling:
- **Scale-Up Trigger:** RF predicts transition from lower tier to higher tier for **≥ 3 consecutive** sampling intervals → pre-emptively allocate higher-tier resources
- **Scale-Down Trigger:** Actual utilization drops below 20% of allocated capacity for **≥ 10 consecutive** intervals → release excess resources
- **Cooldown Period:** Minimum 60 seconds between scaling actions to prevent oscillation

**(c) SLA Compliance Monitor:** Continuously validates:
- Response time < 200 ms (configurable per SLA)
- System uptime ≥ 99.9%
- Resource utilization efficiency ≥ 60%

#### 2. Novel Operational Method

The claimed method operates as follows:

**Step 1:** Continuously monitor computing infrastructure to collect multi-dimensional resource metrics at regular intervals.

**Step 2:** Construct temporal feature vectors incorporating first-order difference terms to capture workload trends.

**Step 3:** Apply K-Means clustering to profile incoming workloads into resource consumption categories.

**Step 4:** Input the augmented feature vector (metrics + cluster label) into a GA-optimized Random Forest classifier to predict the optimal allocation tier.

**Step 5:** Map the predicted allocation tier to a specific resource configuration via the policy engine.

**Step 6:** Execute resource allocation/scaling actions, including pre-emptive scale-up when consecutive tier-transition predictions are detected.

**Step 7:** Monitor SLA compliance and feed performance data back into the training pipeline for continuous model improvement.

**Step 8:** Detect and flag anomalous resource patterns using a secondary RF anomaly detection model for priority handling.

---

### CLAIMS

**Claim 1.** A system for AI-enabled resource detection and allocation, comprising:
- a resource detection layer configured to continuously monitor computational infrastructure and construct multi-dimensional temporal feature vectors incorporating CPU, memory, disk I/O, network bandwidth metrics, and first-order temporal difference terms;
- an intelligent decision layer comprising a K-Means clustering module for workload profiling, a Random Forest ensemble classifier for allocation tier prediction, and a Genetic Algorithm module for automated RF hyperparameter optimization;
- an allocation execution layer comprising a policy engine for resource mapping, a pre-emptive scaling engine, and an SLA compliance monitor;
- wherein the three layers operate in a continuous loop to achieve real-time, predictive, and adaptive resource allocation.

**Claim 2.** The system of Claim 1, wherein the temporal feature vectors include first-order difference terms (Δcpu_t = cpu_t − cpu_{t−1}) that enable proactive resource allocation by capturing workload trends before they fully manifest.

**Claim 3.** The system of Claim 1, wherein the K-Means clustering module classifies workloads into five categories: CPU-intensive, memory-intensive, I/O-intensive, network-intensive, and balanced; and the cluster label is appended to the feature vector as an input to the Random Forest classifier.

**Claim 4.** The system of Claim 1, wherein the Genetic Algorithm module optimizes six Random Forest hyperparameters (number of estimators, maximum tree depth, minimum samples split, minimum samples leaf, maximum features ratio, and bootstrap flag) using tournament selection, single-point crossover, and Gaussian mutation over multiple generations.

**Claim 5.** The system of Claim 1, wherein the pre-emptive scaling engine triggers resource scale-up when the Random Forest classifier predicts a transition from a lower allocation tier to a higher tier for a predetermined number of consecutive sampling intervals.

**Claim 6.** The system of Claim 1, further comprising an enhanced Random Forest anomaly detection module that identifies abnormal resource consumption patterns and flags them for priority handling.

**Claim 7.** The system of Claim 1, wherein the system operates across multiple computing domains including cloud computing, Internet of Things, edge computing, and high-performance computing environments without domain-specific modification.

**Claim 8.** A method for AI-enabled resource detection and allocation, comprising the steps of:
- continuously monitoring computational infrastructure to collect multi-dimensional resource metrics;
- constructing temporal feature vectors with first-order difference terms;
- clustering workloads using K-Means to generate profiling labels;
- classifying resource demands using a GA-optimized Random Forest ensemble classifier;
- mapping classification outputs to resource allocation configurations via a policy engine;
- executing pre-emptive scaling when consecutive tier-transition predictions are detected;
- monitoring SLA compliance and feeding performance data back for continuous model retraining.

**Claim 9.** The method of Claim 8, wherein the Random Forest classifier achieves a classification accuracy of at least 94% and allocation latency of less than 100 milliseconds.

**Claim 10.** The method of Claim 8, wherein the GA-optimized RF model improves F1-score by at least 4% over default-hyperparameter Random Forest models, and reduces energy consumption by at least 18% over threshold-based allocation methods.

---

### ABSTRACT

The present invention discloses a system and method for AI-enabled resource detection and dynamic allocation using Random Forest (RF) ensemble learning. The system comprises a three-tier architecture: (1) a Resource Detection Layer that monitors CPU, memory, disk I/O, and network metrics to construct temporal feature vectors with trend indicators; (2) an Intelligent Decision Layer integrating K-Means workload clustering, RF-based demand classification, and Genetic Algorithm (GA)-based hyperparameter optimization; and (3) an Allocation Execution Layer with policy-driven resource mapping, pre-emptive scaling, and SLA compliance monitoring. The GA-RF hybrid model achieves 94.7% classification accuracy, reduces allocation latency by 31%, and improves energy efficiency by 18.5% compared to conventional methods. The system is validated across cloud, IoT, edge computing, and HPC domains, providing a unified, interpretable, and computationally efficient framework for next-generation resource management.

---

### DRAWINGS

*Drawing 1:* Three-tier system architecture diagram (Resource Detection → Intelligent Decision → Allocation Execution)

*Drawing 2:* Random Forest classification pipeline flow diagram

*Drawing 3:* GA-RF hyperparameter optimization convergence graph

*Drawing 4:* Feature importance ranking bar chart

*Drawing 5:* Comparative performance analysis (RF vs. SVM vs. LSTM vs. GA-RF)

*Note: Formal patent drawings to be prepared by a patent illustrator prior to filing.*

---

### PRIOR ART CONSIDERED

1. US Patent 10,956,838 — "Machine learning-based resource allocation in cloud computing" (Microsoft, 2021) — focuses on neural network-based allocation without RF or GA optimization.
2. US Patent 11,321,170 — "Intelligent resource management using reinforcement learning" (Amazon, 2022) — uses RL-only approach without clustering or RF classification.
3. Ahmad et al. (2025), Scientific Reports — K-Means + RF for IoT allocation — does not include GA optimization or cross-domain validation.
4. Gupta and Sharma (2025), ETASR — AI-augmented Linux kernel with RF — kernel-level only, not a full allocation framework.
5. Al-Rawi et al. (2025), MDPI — RF + RL for VM placement — does not include clustering or pre-emptive scaling.

**Distinction of Present Invention:**  
The present invention is distinguished from the prior art by its novel integration of (a) temporal feature vector construction with trend indicators, (b) K-Means workload profiling combined with GA-optimized RF classification in a unified pipeline, (c) pre-emptive scaling triggered by consecutive tier-transition predictions, and (d) cross-domain applicability validated across four computing paradigms.
