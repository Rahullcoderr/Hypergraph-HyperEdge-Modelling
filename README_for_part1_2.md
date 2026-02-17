# 📘 Part 1 — Hypergraph Neural Network (HGNN) on Cora
### Citation-Based Hypergraph Construction

---

## 📌 Overview

This section implements a **Hypergraph Neural Network (HGNN)** for semi-supervised node classification on the **Cora citation dataset**.

Instead of using a standard graph convolution, we:

1. Convert the citation graph into a hypergraph
2. Construct the incidence matrix
3. Compute the hypergraph propagation matrix
4. Train a 2-layer HGNN

---

## 📂 Dataset: Cora

- Nodes (papers): 2708  
- Edges (citations): ~10k  
- Node features: 1433-dimensional bag-of-words  
- Classes: 7 research topics  

We begin with a simple graph:

G = (V, E')

Where:
- V = set of nodes  
- E' ⊆ V × V = citation edges  

---

# 🔁 Step 1 — From Graph to Hypergraph

A simple graph edge connects exactly two nodes.

A hypergraph allows a hyperedge:

e ⊆ V

where a hyperedge can connect multiple nodes simultaneously.

---

## 🔹 Hypergraph Construction Rule

For each node v_i, we construct one hyperedge:

e_i = { v_i } ∪ N(v_i)

That is:
- The node itself
- All its 1-hop citation neighbors

Thus:

|E| = |V| = N

Each node generates one hyperedge.

---

# 📐 Step 2 — Incidence Matrix

We represent the hypergraph using the incidence matrix:

H ∈ {0,1}^{N × M}

Where:

H[i, j] = 1  if node v_i belongs to hyperedge e_j  
H[i, j] = 0  otherwise

Since each node creates one hyperedge:

M = N

So:

H ∈ R^{N × N}

---

## 🔹 Sparse Representation

The citation network is sparse:

nnz(H) << N²

Therefore, H is stored as a sparse COO tensor for memory efficiency.

---

# 📊 Step 3 — Degree Matrices

### Vertex Degree

d(v_i) = sum over hyperedges containing v_i

Matrix form:

Dv = diag(d(v_1), ..., d(v_N))

---

### Hyperedge Degree

δ(e_j) = number of vertices in hyperedge e_j

Matrix form:

De = diag(δ(e_1), ..., δ(e_M))

---

# 🔄 Step 4 — Hypergraph Propagation Matrix

HGNN uses the propagation operator:

G = Dv^{-1/2} · H · De^{-1} · Hᵀ · Dv^{-1/2}

This corresponds to:

vertex → hyperedge → vertex

---

## 🔹 Interpretation

1. Aggregate vertices into hyperedges:
   Hᵀ X

2. Normalize hyperedge influence:
   De^{-1}

3. Propagate back to vertices:
   H

4. Apply symmetric normalization:
   Dv^{-1/2}

This produces a normalized high-order propagation matrix.

---

# 🧠 Step 5 — HGNN Layer

For layer l:

X^{(l+1)} = σ( G · X^{(l)} · W^{(l)} )

Where:

- X^{(l)} ∈ R^{N × C_l}
- W^{(l)} ∈ R^{C_l × C_{l+1}}
- σ = ReLU activation

---

## 🔹 Model Architecture

2-layer HGNN:

X¹ = ReLU( G · X⁰ · W⁰ )

X² = G · X¹ · W¹

Dropout is applied after the first layer.

---

# 🎯 Step 6 — Training Objective

Semi-supervised cross-entropy loss:

L = - Σ_{i ∈ train} Σ_{c=1}^C y_ic log(ŷ_ic)

Training configuration:

- Optimizer: Adam  
- Learning rate: 0.01  
- Weight decay: 5e-4  
- Early stopping (patience = 50)

---

# 📈 Results

Using the standard Planetoid split, this configuration typically achieves:

- ~80% test accuracy
- Stable convergence
- Smooth validation curves

---

# 💡 Why This Works

The Cora citation graph exhibits homophily:

> Papers tend to cite papers from the same research area.

Thus, hyperedges encode local topic clusters.

HGNN propagation smooths feature representations across citation neighborhoods.

---

# 🧾 Key Insight

This hypergraph construction is mathematically close to:

A + I

(where A is the adjacency matrix)

Thus HGNN behaves similarly to GCN, but through the hypergraph formalism.

---

# 📘 Part 2 — KNN-Based Hypergraph Construction

## 📌 Overview

In Part 1, hyperedges were built using **citation neighbors**.

In this part, we instead construct hyperedges using **feature similarity (K-Nearest Neighbors)**.

Everything else remains the same:
- Same HGNN architecture
- Same propagation formula
- Same training loop
- Same evaluation setup

Only the hypergraph structure changes.

---

## 🔁 Hypergraph Construction

For each node \( v_i \), we compute cosine similarity in feature space:

1. Normalize features:
   x_i' = x_i / ||x_i||₂

2. Compute similarity matrix:
   S = X_norm · X_normᵀ

3. Select top-K most similar nodes:
   e_i = TopK(S[i, :])

Each node generates one hyperedge of size K.

Total hyperedges:
|E| = N

---

## 📊 Incidence Matrix

We construct:

H ∈ {0,1}^{N × N}

Each column contains exactly K ones.

Number of non-zeros:
nnz(H) ≈ N × K

---

## 🔄 Propagation (Same as Part 1)

G = Dv^{-1/2} · H · De^{-1} · Hᵀ · Dv^{-1/2}

HGNN layer:

X^{(l+1)} = σ( G · X^{(l)} · W^{(l)} )

---

## 🧠 Key Difference from Part 1

| Citation Hypergraph | KNN Hypergraph |
|---------------------|----------------|
| Uses graph structure | Uses feature similarity |
| Structural homophily | Geometric proximity |
| Similar to GCN | Similar to manifold smoothing |

---

## 📈 Notes

- Small K → localized smoothing  
- Large K → over-smoothing  
- Performance depends on feature quality  

This variant tests whether feature-space neighborhoods improve node classification.



