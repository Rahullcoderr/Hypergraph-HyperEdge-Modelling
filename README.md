# DOSAGE + HGNN on Cora

This repository implements the DOSAGE hyperedge construction algorithm integrated with a Hypergraph Neural Network (HGNN) for node classification on the Cora dataset.

This implementation follows the algorithm described in:

- DOSAGE: Densest Overlapping Subgraphs via Adaptive Greedy Enumeration
- HGNN: Hypergraph Neural Networks

---

# 1. CTODS vs DOSAGE

The paper defines:

## CTODS (Theoretical Problem)

Constrained Top-K Overlapping Densest Subgraphs:

Maximize:

    r(W) = Σ density(S_i) + λ Σ distance(S_i, S_j)

Subject to:

- α ≤ |S| ≤ β
- diameter ≤ δ

The paper proves CTODS is NP-hard.

This refers to the exact global optimization problem.

---

## DOSAGE (Algorithm 1 in Paper)

DOSAGE is the greedy algorithm proposed to approximately solve CTODS.

It consists of:

1. Extract densest subgraph via greedy peeling.
2. Iteratively extract the next subgraph maximizing the objective.
3. Repeat until k subgraphs are selected.

Important:
The paper explicitly enumerates candidate subgraphs as:

    for subset S ⊆ V, |S| ∈ [α, β]

This is exponential in |V|.

So although DOSAGE is greedy, the candidate search step is still combinatorial.

---

# 2. What Is Implemented in This Repository

We implement:

- Exact density definition
- Exact distance function
- Exact objective function
- Goldberg-style greedy peeling
- Size constraint (α, β)
- Diameter constraint (δ)
- Greedy iterative extraction
- Diversity via objective maximization

The core DOSAGE logic is preserved.

---

# 3. Practical Modification

## Candidate Subgraph Enumeration

Paper specifies:

    for subset S ⊆ V

This is exponential and infeasible on Cora (2708 nodes).

Instead, this implementation replaces exhaustive subset enumeration with:

1. Random seed sampling
2. Local k-hop neighborhood extraction
3. Greedy densest subgraph extraction within that region
4. Objective evaluation
5. Selection of best candidate among sampled trials

This modification:

- Preserves DOSAGE objective
- Preserves constraints
- Preserves greedy framework
- Replaces global enumeration with scalable candidate exploration

This is necessary for computational feasibility.

---

# 4. Hypergraph Construction

From extracted subgraphs W:

- Build incidence matrix H
- Rows = nodes
- Columns = hyperedges
- H(i,j) = 1 if node i ∈ hyperedge j

Sparse tensors are used.

---

# 5. HGNN Propagation

Propagation matrix:

    G = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}

This matches the HGNN paper exactly.

No modification to the propagation rule.

---

# 6. Experimental Results

Current results on Cora:

- Train Accuracy ≈ 47%
- Test Accuracy ≈ 35%

---

# 7. Coverage Analysis

Hyperedge coverage:

- Covered nodes ≈ 728
- Total nodes = 2708

Only ~27% of nodes participate in at least one hyperedge.

For uncovered nodes:

- dv = 0
- Propagation row becomes zero
- Model behaves like MLP for those nodes

This is the main reason for low accuracy.

---

# 8. Why Coverage Is Low

1. Strict size constraints (α, β)
2. Dense-core bias of peeling
3. Sampling-based candidate search
4. Peripheral sparse nodes rarely selected

DOSAGE tends to extract dense cores repeatedly.

---

# 9. Limitations of Current Implementation

- Does not guarantee full node coverage
- Sampling may miss many valid subgraphs
- Hypergraph may be structurally incomplete

The main bottleneck is coverage, not HGNN.

---

# 10. Possible Improvements

- Increase k
- Increase β
- Reduce α
- Increase sampling trials
- Ensure coverage completion (add fallback hyperedges)
- Hybrid strategy (DOSAGE + KNN hyperedges)

---

# 11. Summary

This repository implements:

- DOSAGE greedy framework (Algorithm 1 structure)
- Practical scalable candidate search
- Exact HGNN propagation

Deviation from paper:

- Replaces exhaustive subset enumeration with sampling-based candidate generation

Reason:

- Full subset enumeration is computationally infeasible on Cora
- Necessary for practical scalability

Main limitation:

- Insufficient hyperedge coverage leads to low classification accuracy.
