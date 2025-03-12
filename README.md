#### Installing requirements:
The requirements are straightforward - no fancy packages other than ```numpy``` and ```scipy``` along with their dependencies.
Nonetheless, you can download all necessary packages like so in a new (virtual) environment:

```python
pip install -r requirements.txt
```

There are three python scripts which are available:  ```fsor_l21.py```, ```scfa_l21.py```, and ```scf_for_scfa_l21.py```.

The hierarchy is as follows:
```fsor_l21.py``` <- ```scfa_l21.py``` <- ```scf_for_scfa_l21.py```

```scf_for_scfa_l21.py``` is used to solve a sub-problem generated in ```scfa_l21.py```, which is used to solve the main problem created in ```fsor_l21.py```.

If you choose to perform feature selection with FSOR- $\ell_{2,1}$, you can begin by importing the following:
```python
from fsor_l21 import fsor_l21
```

The function ```fsor_l21``` inside the script ```fsor_l21``` takes in the following inputs:




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
