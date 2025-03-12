#### Requirements:
The requirements are straightforward - no fancy packages other than ```numpy``` and ```scipy``` along with their dependencies.
Nonetheless, you can download all necessary packages like so in a new (virtual) environment:

```python
pip install -r requirements.txt
```

There are three python scripts which are available in the ```src``` folder:  ```fsor_l21.py```, ```scfa_l21.py```, and ```scf_for_scfa_l21.py```.

The hierarchy is as follows:
```fsor_l21.py``` <- ```scfa_l21.py``` <- ```scf_for_scfa_l21.py```

```scf_for_scfa_l21.py``` is used to solve a sub-problem generated in ```scfa_l21.py```, which is used to solve the main problem created in ```fsor_l21.py```.

If you choose to perform feature selection with FSOR- $\ell_{2,1}$, you can begin by importing the following:
```python
from fsor_l21 import fsor_l21
```

The function ```fsor_l21``` inside the script ```fsor_l21.py``` takes in the following inputs:
```python
# X:  n x m mean-centered or standardized data matrix (n x m numpy array)
# Y:  k x m mean-centered or standardized label amtrix (k x m numpy array)
# opts:  class object which can be defined as follows:

class optimize_options:
    def __init__(self):
        self.tol = None # Stopping tolerance for numerical solving algorithm (e.g. 1e-4)
        self.maxit = None # Maximum number of iterations for numerical solving algorithm (e.g. 500)
        self.init = None # 1 if initial W is provided in self.W and 0 if you want initial W to be andomly generated inside fsor_l21
        self.W = None # Initial guess W (assuming self.init = 1)
        self.lambda_param = None # Positive regularization hyperparameter (e.g. 1e-2)

opts = optimize_options()
opts.tol = 1e-6; opts.maxit = 1000; 
opts.lambda_param = 1e-2;
opts.init = 1
opts.W = scipy.linalg.orth(np.random.rand(n, k)) # Creating a random initial guess which has orthonormal columns.

output_info = fsor_l21(X,Y,opts)
```

The output of ```fsor_l21``` is as follows:

```python

output_info.time # CPU time in seconds for FSOR-l21 to solve
output_info.f # List of objective function values over each iteration
output_info.W # Last solved projection matrix W
output_info.wts # List of weights with each entry in between 0 and 1 designating importance to weight with associated index in W
                # The closer a feature's weight is to 1, the more important it is in predicting a datapoint's class.
output_info.res_all # List of normalized KKT residuals over each iteration

``` 



#### BibTeX Citation:
Please cite my work if you'd like to use it in your own.

```latex
@INCOLLECTION{Chairez2024-gs,
  title     = "Orthogonal Single-view and Multi-view Feature Selection Models
               via Spectral Theory Based Methods",
  booktitle = "Mathematics Dissertations",
  author    = "Chairez, Zachary",
  year      =  2024
}
