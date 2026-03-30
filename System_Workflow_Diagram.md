# AIRDA Framework: System Workflow Diagram

This document contains the end-to-end resource detection and allocation pipeline workflow for the **AIRDA Framework**, as described in the research paper.

## End-to-End Processing Pipeline

The pipeline executes the following 7 steps during every monitoring interval:

1. **MONITOR**: Collect Resource Metrics (CPU, Memory, Disk, Network, Task Queue, and temporal trends).
2. **DETECT**: Anomaly Identification using Enhanced RF.
3. **PROFILE**: K-Means Workload Clustering to assign a cluster label ($c_j$).
4. **CLASSIFY**: GA-RF Demand Prediction to predict the allocation tier (Low, Medium, High, Critical).
5. **ALLOCATE**: Policy-Driven Resource Mapping to allocate physical resources based on the tier.
6. **SCALE**: Dynamic Pre-emptive Scaling to scale up/down based on temporal patterns.
7. **VALIDATE**: SLA Compliance & Feedback Loop.

## Workflow Flowchart

```mermaid
flowchart TD
    Start([Start Monitoring Cycle]) --> M[Step 1: MONITOR<br/>Collect 9D Resource Metrics x_t]
    
    M --> D{Step 2: DETECT<br/>Anomaly Score > α?}
    
    D -- Yes --> Crit[Flag for Priority<br/>Escalate to Critical Tier]
    Crit --> A
    
    D -- No --> P[Step 3: PROFILE<br/>K-Means Workload Clustering<br/>Assign cluster c_j]
    
    P --> C[Step 4: CLASSIFY<br/>GA-RF Classifier Predicts Tier ŷ_t<br/>using x'_t = x_t + c_j]
    
    C --> A[Step 5: ALLOCATE<br/>Map ŷ_t to Resource Policy<br/>Low/Med/High/Critical]
    
    A --> S{Step 6: SCALE<br/>Scaling Conditions Met?}
    
    S -- Upward Trend<br/>>3 Intervals --> ScaleUp[Trigger Pre-emptive Scale-Up]
    S -- Util < 20%<br/>>10 Intervals --> ScaleDown[Trigger Scale-Down]
    S -- Stable --> V[Step 7: VALIDATE<br/>Check SLA Compliance]
    
    ScaleUp --> V
    ScaleDown --> V
    
    V --> SLA{SLA Violated?<br/>Resp > 200ms or <br/>Uptime < 99.9%}
    
    SLA -- Yes --> Log[Log Violation & Adjust Tier Upward]
    Log --> Loop
    SLA -- No --> Loop
    
    Loop[Feedback Loop] --> M
    
    %% Styling
    classDef step fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1;
    classDef decision fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#bf360c;
    classDef action fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20;
    classDef anomaly fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#b71c1c;
    classDef startnode fill:#fafafa,stroke:#424242,stroke-width:2px,color:#212121;
    
    class Start startnode;
    class M,P,C step;
    class D,S,SLA decision;
    class A,ScaleUp,ScaleDown,V,Loop action;
    class Crit,Log anomaly;
```
