# Cost-aware Runtime Framework Specification

<!-- LEGACY-NON-AUTHORITATIVE: Phase X X0.5 -->

**Date**: 2026-07-21  
**Phase**: 7 Step 5  
**Status**: Draft  

## 1. Overview
This document specifies the **Unified Runtime Framework** for MuKG, evolved from the Cost-aware Batch Packing (CBP) and GPU cost modeling phases. It defines modules, interfaces, data flow, and extension points. The framework abstracts execution costs, scheduling, and hardware backends, enabling systematic optimization of knowledge graph embedding training.

## 2. Unified Runtime Cost Model
The cost model is no longer a simple predictor of negative sampling time. It becomes a **universal runtime abstraction** that maps a triple or a batch to a scalar cost value, interpretable across CPU, GPU, and future backends.

**Current implementation**:  
`cost(triple) = E_retry * B3_const`, where `E_retry` depends on candidate size (from Phase 5.6). This cost correlates with CPU sampling time.

**GPU generalization**:  
For GPU execution, cost can be redefined as:
- **Base cost**: fixed per batch (kernel launch, data transfer).
- **Variable cost**: proportional to number of negatives, or `batch_size * neg_num`.
- **Advanced cost**: can incorporate memory footprint, communication volume (DDP), or compute intensity.

**Interface**:
```python
class CostModel:
    def predict_triple_cost(triple) -> float
    def predict_batch_cost(batch_triples) -> float
```

**Extension**:  
Replace internal logic to use GPU‑relevant features (e.g., entity degree for collision probability) without changing the scheduler/policy.

## 3. Runtime Policy
The Runtime Policy is the decision layer that determines *how* a batch is executed. It abstracts over hardware topologies and execution strategies.

**Current default**: `GPUOnly` – all batches run GPU negative sampling.

**Future policies**:
- `SingleGPU` – default execution.
- `DDP` – balanced batches to minimize synchronization wait.
- `PipelineParallel` – split micro‑batches across devices.
- `MemoryPool` – reuse pre‑allocated tensors.
- `Overlap` – hide data transfer behind computation.

**Interface**:
```python
class RuntimePolicy:
    def decide(batch) -> ExecutionStrategy
```

The policy receives batch metadata (cost, size) and returns an execution descriptor.

## 4. Framework Architecture
```
┌──────────────────────────────────────────────────┐
│              Cost-aware Runtime Framework         │
│                                                   │
│  ┌───────────────────┐                            │
│  │ Runtime Cost Model│  (per‑triple/batch cost)   │
│  └────────┬──────────┘                            │
│           │                                        │
│  ┌────────▼──────────┐                            │
│  │ Runtime Scheduler │  (CostSorter + FFDPacker)  │
│  └────────┬──────────┘                            │
│           │                                        │
│  ┌────────▼──────────┐                            │
│  │ Runtime Policy    │  (ExecutionDecision)       │
│  └────────┬──────────┘                            │
│           │                                        │
│  ┌────────▼──────────┐                            │
│  │ GPU Execution     │  (GPUSampler, forward/back)│
│  └────────┬──────────┘                            │
│           │                                        │
│  ┌────────▼──────────┐                            │
│  │ Batch Provider    │  (zero‑intrusion adapter)  │
│  └───────────────────┘                            │
└──────────────────────────────────────────────────┘
```

## 5. Module Interfaces

### 5.1 CostModel
- **Input**: triple (h, r, t) or list of triples.
- **Output**: scalar cost.
- **Dependencies**: entity degree, relation frequency (cached).

### 5.2 Scheduler
- **Input**: list of triples, cost table.
- **Output**: ordered/packed list of batches.
- **Strategy**: pluggable sorters (Random, Cost) and packers (Chunk, FFD).

### 5.3 RuntimePolicy
- **Input**: batch metadata (cost, batch size, device affinity).
- **Output**: execution strategy object.

### 5.4 GPUExecution
- **Input**: batch of positive triples.
- **Output**: negative sample tensors (on GPU).
- **Implementation**: `torch.randint` + `torch.isin` for tail corruption; extensible to head corruption.

### 5.5 BatchProvider
- **Input**: scheduler, dataset.
- **Output**: iterable of batches (positive triples).
- **Role**: replaces PyTorch DataLoader; triggers scheduler per epoch.

## 6. Data Flow (per epoch)
```
1. Feature extraction (entity degrees) → CostModel
2. Scheduler.pack_batches(triples, cost_table) → list of batch triples
3. For each batch:
   a. BatchProvider yields batch_triples
   b. GPUSampler.generate(batch_triples) → neg_heads, neg_tails
   c. Forward/backward (model)
   d. Optimizer step
```

## 7. Extension Points
The framework is designed to accommodate new modules without altering core interfaces:

| Extension | Insertion Point | Effort |
|-----------|----------------|--------|
| GPU‑based Cost Model | Replace `CostModel.predict_triple_cost` | Low |
| Alternative packers (e.g., round‑robin) | Add new packer class in `Scheduler` | Low |
| Multi‑GPU policy | Implement new `RuntimePolicy` subclass | Medium |
| Asynchronous data transfer | Modify `GPUExecution` to use CUDA streams | Medium |
| Negative caching | Insert cache layer before `GPUExecution` | Medium |
| Mixed precision support | Wrap in `RuntimePolicy` or `GPUExecution` | Low |

## 8. Next Steps
- **Phase 7 Step 6**: Translate this specification into a concrete implementation plan (break down into Phase 8 nodes).
- **Phase 8**: Implement GPU Sampler, integrate with CBP, and run end‑to‑end evaluation (single GPU, then DDP).

This specification will be validated during Phase 8 implementation and revised as needed.
