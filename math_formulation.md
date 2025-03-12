#### Supervised Feature Selection via Orthogonal Regression and $\ell_{2,1}$-norm Regularization (FSOR- $\ell_{2,1}$)

FSOR- $\ell_{2,1}$ is defined through the following optimization problem:

$$
\min_{W \in \mathbb{R}^{n \times k}} \lVert W^TX - Y \rVert_F^2 + \lambda \lVert W \rVert_{2,1} 
\quad \text{s.t.} \quad W^TW = I_k, 
$$

where $X \in \mathbb{R}^{n \times m}$ is a mean-centered data matrix, $Y \in \mathbb{R}^{k \times m}$ is a mean-centered one-hot encoded label matrix, $\lambda > 0$ is a regularization hyperparameter, $I_k$ is the identity matrix of size $k \times k$, and 

$$ 
\lVert W \rVert_{2,1} = \sum_{i = 1}^n \lVert \boldsymbol{w}^i \rVert_2, 
$$

where $\boldsymbol{w}^i$ is the $i^{\text{th}}$ row of $W$.

We can reformualte FSOR- $\ell_{2,1}$ as a Eigenvector Dependent Nonlinear Eigenvalue Problem (NEPv):

Find $W \in \mathbb{R}^{n \times k}$ such that $W^TW = I_k$ and 

$$
  J(W)W = W \Phi, 
$$

where $J(W) = A + BW^T + WB^T + DWW^T + WW^TD$, $\Phi = W^T J(W) W$, $A = XX^T$, $B = -XY^T$, $D$ is a diagonal matrix whose $i^{\text{th}}$ diagonal entry is equal to $(2 \lVert \boldsymbol{w}^i \rVert_2)^{-1}$, and $W$ is an orthonormal eigenbasis matrix of $J(W)$ associated with its $k$ smallest eigenvalues.  

#### Numerical Method to Solve FSOR- $\ell_{2,1}$
We solve the NEPv form of FSOR- $\ell_{2,1}$ with a refined, accelerated version of the basic Self Consistent Field Iteration.
