# 3. Method

This section presents our cost-aware unified runtime framework for efficient KGE training. We first profile the standard training pipeline to identify the dominant bottleneck (Section 3.1). Motivated by the profiling results, we introduce an offline cost model that predicts per-triple negative sampling cost from static knowledge graph statistics (Section 3.2). We then describe Cost-aware Batch Packing (CBP), a pluggable sort-and-pack scheduling layer that leverages the cost model to reduce inter-batch variance on CPU (Section 3.3). Observing that CPU-side overheads limit the benefit of scheduling alone, we propose a fully vectorized GPU-native negative sampling kernel (Section 3.4). Finally, we unify these components into a modular runtime framework that enables transparent CPU-to-GPU migration while preserving extensibility (Section 3.5).

---

## 3.1 Background and Profiling Analysis

### 3.1.1 KGE Training Loop

Knowledge graph embedding (KGE) models learn low-dimensional vector representations for entities and relations by optimizing a scoring function $f(h, r, t)$ over a set of observed triples $\mathcal{T}$. A standard training iteration for a batch $\mathcal{B} \subset \mathcal{T}$ proceeds as follows:

1. **Collate**: Retrieve entity and relation IDs for each triple in $\mathcal{B}$.
2. **Negative Sampling**: For each positive triple $(h, r, t) \in \mathcal{B}$, generate $N$ negative triples $(h', r, t)$ or $(h, r, t')$ by corrupting either the head or tail entity, ensuring that the resulting triple does not appear in $\mathcal{T}$.
3. **Tensor Construction**: Convert the collated indices and negative samples into GPU tensors for forward computation.
4. **Forward/Backward**: Compute scores $f(h, r, t)$ for positive and negative triples, evaluate the margin-based ranking loss, and perform backpropagation.
5. **Optimizer Step**: Update model parameters.

### 3.1.2 Profiling Motivation

We profiled the above loop on FB15k-237 using TransE with batch size 5,000 and 150 negative samples per positive triple. As shown in Fig. 1, negative sampling consumes $35.7\%$ of total step time—when combined with collation ($46.6\%$), data preparation accounts for over $82\%$ of the training step, dwarfing the forward/backward computation ($7.0\%$).

Further instrumentation reveals that the negative sampling cost per batch is highly variable: the standard deviation of per-batch sampling time is $28.5$ ms at batch size $5{,}000$ (the full-loop standard deviation reaches $34.3$ ms). This variance is primarily driven by the collision-retry loop in CPU-side negative sampling. Formally, each negative sample requires:

- **B1 (Random Sampling)**: Draw a candidate entity uniformly from $\{1, \dots, |\mathcal{E}|\}$.
- **B2 (Candidate Construction)**: Form the candidate negative triple.
- **B3 (Collision Check)**: Verify that the candidate triple $\notin \mathcal{T}$ (the set of all known true triples).
- **B4 (Retry)**: If collision occurs, repeat B1–B3 up to a maximum of $K$ attempts.

The expected number of retries for entities with large candidate pools (high-connectivity "hub" entities) follows a geometric distribution, causing per-batch sampling time to fluctuate significantly. Empirically, the correlation between entity degree and measured sampling cost reaches $R = 0.816$ (see Section 4.4), confirming that structural properties of the knowledge graph directly govern runtime behavior.

**Motivation for our approach**: Existing KGE training frameworks treat negative sampling as an opaque preprocessing step with no cost awareness. No existing system exploits the predictable relationship between graph topology and sampling cost to inform batch scheduling or hardware allocation. Our framework addresses this gap through three pillars: (1) an offline cost model that predicts sampling cost from static KG features, (2) a cost-aware batch scheduler that reduces inter-batch variance, and (3) a GPU-native sampler that eliminates the CPU bottleneck entirely.

---

## 3.2 Offline Cost Model

### 3.2.1 Intuition

The core insight of our cost model is that the expected negative sampling cost of a triple $(h, r, t)$ is predominantly determined by the *candidate size* of the corrupted entity—i.e., the number of entities sharing the same relation-type connectivity pattern. When corrupting the tail entity, all entities that have appeared as the tail of relation $r$ form the effective candidate pool; the collision probability during B3 is proportional to the size of this pool relative to the full entity set.

Formally, for a given entity $e$, we define its *candidate size* $c_e$ as:

$$
c_e = \max(|\{e' : (e', r, \cdot) \in \mathcal{T}\}|, |\{e' : (\cdot, r, e') \in \mathcal{T}\}|)
$$

where the maximum is taken over all relations incident to $e$. This captures the worst-case candidate pool size when $e$ is selected as the entity to be corrupted.

### 3.2.2 Cost Prediction

The expected negative sampling time for a triple can be expressed as:

$$
\mathbb{E}[T_{\text{neg}}] = \mathbb{E}[N_{\text{retry}}] \cdot t_{\text{B3}}
$$

where $t_{\text{B3}}$ is the constant cost of one collision check (set membership query), and $\mathbb{E}[N_{\text{retry}}]$ is the expected number of retry attempts. Under the uniform corruption model, the collision probability for one candidate is $p_{\text{coll}} = c / |\mathcal{E}|$, and assuming independent retries (a reasonable approximation when $c \ll |\mathcal{E}|$):

$$
\mathbb{E}[N_{\text{retry}}] = \frac{1}{1 - p_{\text{coll}}} = \frac{|\mathcal{E}|}{|\mathcal{E}| - c}
$$

For a knowledge graph with highly skewed degree distributions, $c$ varies dramatically across entities. Head entities in FB15k-237 have degree ranging from 1 to over $10^4$, yielding collision probabilities that span two orders of magnitude. Consequently, per-triple negative sampling cost can differ by up to $100\times$, making it imperative for the runtime to be *cost-aware*.

### 3.2.3 Offline Model Fitting

We fit a linear regression model mapping $c_e$ to the *measured* negative sampling time for entity $e$ (averaged over all triples where $e$ appears as the corrupted entity). Using $455$ sampled entities from FB15k-237, we obtain:

$$
\hat{T}_{\text{neg}}(c) = \alpha \cdot c + \beta
$$

with $R^2 = 0.9008$, indicating that candidate size alone explains $90\%$ of the variance in sampling cost. The fitted parameters are pre-computed offline once per dataset and stored as a lookup table $\mathbf{C} \in \mathbb{R}^{|\mathcal{E}|}$ where $\mathbf{C}[e] = \hat{T}_{\text{neg}}(c_e)$. This table requires negligible memory ($\approx 116$ KB for $14{,}505$ entities at 64-bit precision) and introduces zero runtime overhead.

**Key property**: The cost model is a pure function of static graph statistics—it requires no online profiling, no model-specific features, and no training. The same cost table can be reused across different KGE models (TransE, RotatE, ConvE, etc.) as long as the negative sampling procedure remains unchanged. This decoupling from model architecture is central to the extensibility of our unified runtime framework.

---

## 3.3 Cost-aware Batch Packing (CBP)

### 3.3.1 Problem Formulation

Given the per-entity cost table $\mathbf{C}$, the total expected cost of a batch $\mathcal{B}$ is:

$$
\Phi(\mathcal{B}) = \sum_{(h, r, t) \in \mathcal{B}} \big(\mathbf{C}[h] + \mathbf{C}[t]\big)
$$

where we sum over both head and tail entities because either may be corrupted during negative sampling (Bernoulli with probability $0.5$). Since all triples in a batch must wait for the slowest negative sampling operation to complete, high inter-batch cost variance directly translates to inefficient GPU utilization and unpredictable training throughput.

The goal of Cost-aware Batch Packing (CBP) is to partition the training triples into batches $\{\mathcal{B}_1, \dots, \mathcal{B}_M\}$ such that the batch cost variance is minimized. This is a classic bin-packing problem with batch size constraints, which we address through a two-stage sorting-and-packing pipeline.

### 3.3.2 Sort-and-Pack Pipeline

Our CBP scheduler defines two pluggable strategies:

**Sort Strategy**. Entities are ordered by a user-defined criterion. We provide two implementations:
- `RandomSorter` (baseline): entities are randomly shuffled, equivalent to standard training batching.
- `CostSorter`: entities are sorted in decreasing order of $\mathbf{C}[e]$. This groups high-cost entities together, enabling the subsequent packing stage to balance batch costs.

**Pack Strategy**. Batches are formed by filling a fixed-size bin (batch size $B$). We provide:
- `ChunkPacker` (baseline): simply chunks the sorted list sequentially.
- `FFDPacker`: applies First-Fit Decreasing (FFD) bin-packing. The first $k \cdot B$ entities (where $k = \lceil|\mathcal{T}| / B\rceil$) are processed as a window; within each window, entities are assigned to bins in order, each entity placed in the first bin with sufficient remaining capacity. This heuristic achieves an asymptotic worst-case ratio of $\frac{11}{9} \text{OPT}$.

The full CBP pipeline is:

```
FeatureExtractor  →  CostModel  →  Scheduler  →  BatchProvider
   (static)          (lookup)     (Sort+Pack)     (iteration)
```

Algorithm 1 formalizes the FFD packing procedure.

```
\begin{algorithm}[ht]
\caption{First-Fit Decreasing Batch Packing}
\label{alg:ffd}
\begin{algorithmic}[1]
\REQUIRE Entity costs $\mathbf{C} \in \mathbb{R}^{|\mathcal{E}|}$, training triples $\mathcal{T}$, batch size $B$
\ENSURE Batches $\{\mathcal{B}_1, \dots, \mathcal{B}_M\}$ with balanced $\Phi(\mathcal{B}_i)$

\STATE Sort entities by decreasing $\mathbf{C}[e]$ (CostSorter)
\STATE Partition sorted entities into windows of size $k \cdot B$ where $k = \lceil|\mathcal{T}| / B\rceil$
\FOR{each window $W$}
    \STATE Initialize $k$ empty bins, each with capacity $B$
    \FOR{each entity $e \in W$}
        \STATE Collect all incident triples $\mathcal{T}_e = \{(h, r, t) \in \mathcal{T} : h = e \lor t = e\}$
        \FOR{each triple $\tau \in \mathcal{T}_e$}
            \STATE Find first bin $j$ with remaining capacity $\geq 1$
            \STATE Assign $\tau$ to bin $j$; decrement bin $j$ capacity
        \ENDFOR
    \ENDFOR
\ENDFOR
\RETURN All bin contents as batches $\{\mathcal{B}_1, \dots, \mathcal{B}_M\}$
\end{algorithmic}
\end{algorithm}
```

### 3.3.3 CBP's Role and Limitations

In controlled experiments on CPU, CBP reduces the coefficient of variation (CV) of batch costs from $0.055$ to $0.012$ (batch size $1{,}000$, $275$ batches per epoch). The standard deviation of per-batch negative sampling time drops by $78\%$ (from $15.5$ ms to $3.4$ ms). This demonstrates that cost-aware scheduling *can* be effective when the system provides sufficient scheduling granularity (many small batches).

However, under standard training conditions (batch size $5{,}000$, only $55$ batches per epoch), CBP's benefit becomes marginal: the negative sampling standard deviation decreases by merely $8.4\%$ ($29.5$ ms to $27.0$ ms). The coarser granularity leaves insufficient room for FFD to balance costs, and other noise sources (Python GIL serialization, OS scheduling jitter, tensor construction variance) dominate the overall runtime. This finding motivates a fundamental shift: rather than scheduling *around* the bottleneck, we should *eliminate* it by migrating negative sampling to the GPU.

---

## 3.4 GPU Runtime Pipeline

### 3.4.1 Design Rationale

The profiling analysis in Section 3.1 reveals that CPU-side negative sampling is bound by two systemic factors: (a) the Python Global Interpreter Lock (GIL), which serializes all sampling operations, and (b) the inherently sequential per-triple for-loop with collision detection. Neither factor can be overcome through CPU-side optimization alone—the fundamental solution is to move the entire sampling operation to the GPU, where massive parallelism replaces sequential iteration.

### 3.4.2 GPU-Native Negative Sampling

Our GPU sampler (Algorithm 2) performs the following operations entirely on-device:

1. **Batch-Level Random Index Generation**: For a batch of $B$ positive triples, generate $B \times N$ random entity indices in a single `torch.randint` call. This replaces the $B \times N$ sequential calls to Python's `random.sample` in the CPU path.

2. **Tail-Only Corruption**: To maintain consistency with the Bernoulli($0.5$) strategy of the CPU sampler while leveraging GPU parallelism, we fix the corruption target to the tail entity. This yields $B \times N$ candidate negative triples in a single tensor operation.

3. **GPU Collision Detection**: Apply `torch.isin` to the candidate triples against the known triple set $\mathcal{T}$. This CUDA-accelerated membership check processes all $B \times N$ candidates simultaneously.

4. **Retry with Masking**: Identify collided candidates via a boolean mask, generate replacement indices only for those positions (`torch.randint` with `masked_fill`), and iterate until no collisions remain (or a maximum retry count is reached).

```
\begin{algorithm}[ht]
\caption{GPU-Native Negative Sampling}
\label{alg:gpu_sampler}
\begin{algorithmic}[1]
\REQUIRE Batch triples $\mathcal{B} = \{(h_i, r_i, t_i)\}_{i=1}^B$, entity count $|\mathcal{E}|$, negative count $N$, known triple set $\mathcal{T}$
\ENSURE Negative head/tail tensors $\mathbf{H}_{\text{neg}}, \mathbf{T}_{\text{neg}} \in \mathbb{N}^{B \times N}$

\STATE $\mathbf{H} \gets [h_1, \dots, h_B]$ \COMMENT{positive heads on GPU}
\STATE $\mathbf{R} \gets [r_1, \dots, r_B]$ \COMMENT{positive relations}
\STATE $\mathbf{T}_{\text{pos}} \gets [t_1, \dots, t_B]$ \COMMENT{positive tails}

\STATE $\mathbf{C} \gets \texttt{randint}(0, |\mathcal{E}|, (B \times N))$ \COMMENT{candidate tail indices}
\STATE $\mathbf{M} \gets \texttt{ones}(B \times N, \texttt{dtype=bool})$ \COMMENT{validity mask}

\FOR{$k = 1$ \TO $K_{\max}$}
    \STATE $\mathbf{C}_{\text{masked}} \gets \mathbf{C}[\mathbf{M}]$ \COMMENT{only unresolved positions}
    \STATE $\mathbf{H}_{\text{cand}} \gets \mathbf{H}.\texttt{repeat\_interleave}(N)[\mathbf{M}]$
    \STATE $\mathbf{R}_{\text{cand}} \gets \mathbf{R}.\texttt{repeat\_interleave}(N)[\mathbf{M}]$

    \STATE \texttt{collision} $\gets \texttt{isin}((\mathbf{H}_{\text{cand}}, \mathbf{R}_{\text{cand}}, \mathbf{C}_{\text{masked}}), \mathcal{T})$
    \STATE $\mathbf{M}[\mathbf{M}] \gets \neg\texttt{collision}$ \COMMENT{update mask: accept non-colliding}
    \IF{$\texttt{sum}(\mathbf{M}) = B \times N$}
        \STATE \textbf{break}
    \ENDIF
    \STATE $\mathbf{C}[\neg\mathbf{M}] \gets \texttt{randint}(0, |\mathcal{E}|, (\texttt{sum}(\neg\mathbf{M}),))$ \COMMENT{replace collided}
\ENDFOR

\STATE $\mathbf{H}_{\text{neg}} \gets \mathbf{H}.\texttt{repeat\_interleave}(N).\texttt{reshape}(B, N)$
\STATE $\mathbf{T}_{\text{neg}} \gets \mathbf{C}.\texttt{reshape}(B, N)$
\RETURN $\mathbf{H}_{\text{neg}}, \mathbf{T}_{\text{neg}}$
\end{algorithmic}
\end{algorithm}
```

### 3.4.3 Performance Characteristics

The GPU sampler achieves three fundamental improvements over its CPU counterpart:

- **Latency**: Negative sampling time drops from $\sim\!600$ ms to $\sim\!3.0$ ms per batch ($198\times$ acceleration). This is a direct consequence of replacing $B \times N = 7.5 \times 10^5$ sequential Python calls with a single CUDA kernel launch.
- **Variance**: The standard deviation of per-batch sampling time collapses from $28.5$ ms to $0.2$ ms ($142\times$ compression). GPU execution is deterministic given fixed inputs; the tiny remaining variance arises solely from CUDA kernel scheduling jitter.
- **Memory**: The GPU sampler requires only $\approx 2$ MB additional VRAM for intermediate tensors (candidate indices and collision masks), negligible on modern GPUs with $8+$ GB memory.

These properties produce a virtuous cycle: eliminating the dominant variance source means the scheduler no longer needs to compensate for unpredictable per-batch costs. The CBP layer, which proved marginal under CPU variance, becomes a lightweight optional component on the GPU path—preserved for extensibility but no longer critical for throughput stability.

---

## 3.5 Unified Runtime Framework

### 3.5.1 Architecture Overview

We integrate the cost model, batch scheduler, and GPU sampler into a unified runtime framework designed around three principles:

1. **Separation of concerns**: Cost prediction, scheduling, and execution are implemented as independent, composable modules.
2. **Offline pre-computation**: All cost information is derived from static KG statistics and cached before training begins.
3. **Transparent hardware migration**: The same scheduler and batch provider operate identically whether the negative sampling backend is CPU or GPU.

Figure X (architecture diagram) illustrates the five-layer pipeline:

```
┌─────────────────────────────────────────────────────────┐
│                   Training Loop                          │
│  model(data) → loss.backward() → optimizer.step()       │
└──────────────────────┬──────────────────────────────────┘
                       │ batches
┌──────────────────────▼──────────────────────────────────┐
│              4. Batch Provider (Adapter)                 │
│  Wraps scheduled batches into PyTorch DataLoader iter    │
│  Enables drop-in replacement of existing loaders         │
└──────────────────────┬──────────────────────────────────┘
                       │ scheduled batch list
┌──────────────────────▼──────────────────────────────────┐
│              3. Scheduler (Sort + Pack)                  │
│  CostSorter / RandomSorter  +  FFDPacker / ChunkPacker   │
│  Produces cost-balanced batch assignments                │
└──────────────────────┬──────────────────────────────────┘
                       │ entity costs
┌──────────────────────▼──────────────────────────────────┐
│              2. Cost Model (Pure Function)               │
│  entity features → candidate_size → expected_cost        │
│  Pre-computed lookup table C ∈ ℝ^{|E|}                   │
└──────────────────────┬──────────────────────────────────┘
                       │ entity features
┌──────────────────────▼──────────────────────────────────┐
│           1. Feature Extractor (Offline)                 │
│  KG statistics: entity degree, relation co-occurrence    │
│  Cached as entity_features.npz                           │
└─────────────────────────────────────────────────────────┘

                    GPU Execution Path
┌─────────────────────────────────────────────────────────┐
│              5. GPU Negative Sampler                     │
│  torch.randint + torch.isin + masked retry               │
│  Completely eliminates CPU for-loop bottleneck           │
└─────────────────────────────────────────────────────────┘
```

### 3.5.2 Module Interfaces

**Feature Extractor** (Layer 1). Traverses the training triple set $\mathcal{T}$ once to compute per-entity statistics: degree in head and tail positions, relation co-occurrence counts, and candidate pool sizes. Output is serialized as a NumPy archive for reuse across experiments. Runtime: $O(|\mathcal{T}|)$, executed once per dataset.

**Cost Model** (Layer 2). Implements the mapping $e \mapsto \mathbf{C}[e]$ described in Section 3.2. This is a pure function with no trainable parameters: given entity features, it evaluates the linear regression and returns a cost estimate in milliseconds. The lookup is $O(1)$ per query and requires no GPU computation.

**Scheduler** (Layer 3). Exposes a `configure(sorter, packer)` interface accepting any combination of sort and pack strategies. The `schedule(triples, cost_table, batch_size)` method returns a list of batch index arrays. Current implementations: $\{\text{Random, Cost}\} \times \{\text{Chunk, FFD}\} = 4$ combinations, extensible to new strategies (e.g., learned sort keys, integer programming packers).

**Batch Provider** (Layer 4). Adapts the scheduler's output to the PyTorch `DataLoader` interface. Wraps `schedule()` output into an iterable that yields `(pos_triples, neg_sampler)` pairs per batch. This adapter pattern allows the framework to be integrated into existing training scripts with minimal code changes.

**GPU Sampler** (Layer 5). A drop-in replacement for the CPU sampler, implementing the `generate(batch_triples) → (neg_heads, neg_tails)` interface. Internally uses PyTorch tensor operations exclusively, requiring no Python-level loop over triples. The sampler is instantiated once at training start with fixed parameters (entity count, negative number) and reused for all epochs.

### 3.5.3 Data Flow and Extensibility

The framework processes a training epoch as follows:

1. **Pre-training** (offline): Feature Extractor runs once → Cost Model builds lookup table → Scheduler computes batch assignments. All results are cached.
2. **Per-epoch** (online): Batch Provider iterates over cached batch assignments → for each batch, GPU Sampler generates negatives → training loop computes loss and updates model.
3. **Configuration switching**: To change from CPU to GPU path, the user replaces the sampler object passed to the Batch Provider; all other layers remain identical. To change scheduling strategy, the user reconfigures the Scheduler's sort/pack parameters; the Cost Model and Batch Provider are unaffected.

This modular design supports several extension points. **Multi-GPU training** can be accommodated by adding a `DeviceSorter` that groups high-cost batches onto specific GPUs while balancing the overall load. **Asynchronous data transfer** can be layered between the Batch Provider and the training loop via CUDA streams, overlapping data preparation with forward computation. **Learned cost models** (neural networks predicting $T_{\text{neg}}$ from richer entity features) can replace the linear regression in Layer 2 without modifying any downstream module—the only requirement is that the replacement exports the same $\mathbf{C}[e]$ lookup interface.

In summary, the Unified Runtime Framework transforms KGE training from a black-box data loading pipeline into a transparent, cost-aware, and hardware-adaptive system. The following section (Section 4) empirically validates each component and quantifies the end-to-end improvements on standard benchmarks.