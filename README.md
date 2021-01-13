# Chimera: hierarchy based multi-objective optimization

``Chimera`` is a general purpose achievement scalarizing function for multi-objective optimization. It allows 
the user to set a hierarchy of objectives along with relative or absolute thresholds for them to be optimized
concurrently. For more details, please refer to the following publication:

F. Häse, L.M. Roch, and A. Aspuru-Guzik. "[Chimera: enabling hierarchy based multi-objective optimization 
for self-driving laboratories](https://pubs.rsc.org/ko/content/articlelanding/2018/sc/c8sc02239a#!divAbstract)". 
*Chemical Science* **2018**, 9(39), 7642-7655.

###  Installation
``Chimera`` can be installed with ``pip``:

```
pip install chimera
```

The installation requires only ``python >= 3`` and ``numpy``.

### Usage

```
from chimera import Chimera

chimera = Chimera(thresholds=[0.5, 0.2], absolute=False, goals=['min', 'max'])
```

In the example above, we have 2 objectives. We want to minimize the first and maximize the second. We use relative 
thresholds (``absolute=False``) to define at which point the next objective in the hierarchy should be optimized. In
this specific example, we are saying that we allow the first objective to degrade up to 50% in order to improve upon
the second objective. And we allow the second objective to degrade up to 20% in order to keep improving the first 
objective further. If we were to set ``thresholds=[0.5, 0.0]``, in the regions of parameters space where the first 
contraint is satisfied (i.e. first objective is better than 50% of the objective values range observed), ``Chimera``
would scalarize the objectives in such a way that we would be optimising only the second objective and never return to 
the first. Please refer to the paper for a more detailed description of the algorithm.

Sometimes we know our objective well, and we might want to optimize to at least a specific value for this objective. 
This setup can be achieved by using absolute rather than relative thresholds, by specifying ``absolute=True``:

```
chimera = Chimera(thresholds=[10, 120], absolute=False, goals=['min', 'max'])
```

In this case, we are saying that we would like to minimize the first objective to at least a value of 10, and maximize
the second objective to at least a value of 120. Where a value of 120 is reached for the second objective, ``Chimera``
will scalarize the objectives in such a way that the first objective becomes again the limiting one, and we will
keep minimizing it as much as possible. To maximize the second objective without any bounds, you can set its absolute
threshold to a very large/small value, even if it may not be reachable in practice. 

Once you have an instance of ``Chimera``, you can use it to scalarize the objective function values obtained to reduce 
them to a single merit value:

```
merit = chimera.scalarize(objectives)
```

Where ``objectives`` is a two-dimensional array with the objective function values for all samples and objectives. Each
row should correspond to a different sample, and each column to a different objective. Note that the order of columns
should reflect the desired hierarchy of the objectives, with the first column being the most important objective and
the last column the least important one.

###  Citation
``Chimera`` is research software. If you make use of it in scientific publications, please cite the following article:

```
@article{chimera,
    author ="Florian Häse and Loïc M. Roch and Alán Aspuru-Guzik",
    title  ="Chimera: enabling hierarchy based multi-objective optimization for self-driving laboratories",
    journal  ="Chemical Science",
    year  = "2018",
    volume  = "9",
    issue  = "39",
    pages  = "7642-7655",
    publisher = "The Royal Society of Chemistry",
    doi = "10.1039/C8SC02239A",
    url = "http://dx.doi.org/10.1039/C8SC02239A"}
```

###  License
``Chimera`` is distributed under an MIT License.
