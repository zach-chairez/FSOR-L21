### **Purpose:**  
This folder contains .py scripts which, together, solve the single-view feature selection model FSOR $\ell_{2,1}$

### **FSOR $\ell_{2,1}$:**
The FSOR $\ell_{2,1}$ feature selection model is defined as the following:

  $$\min_{W \in \mathbb{R}^{n \times k}} \ f(W) \coloneqq \text{trace}(W^T A W) + 2\text{trace}(W^TB) + \lambda \lVert W \rVert_{2,1}
    \ \text{such that} \ W^TW = I_k$$/
