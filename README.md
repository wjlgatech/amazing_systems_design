# ML systems Design

1. Project setup: understand the problem deep and wide enough for [end user, business, ML system, ...]
  - Goals: what difference the ML will make before & after
  - Constraints: how fast, how good the predictions need to be? The cost of false positive VS false negative?
  - Evaluation
  - UX: user experience walk through
    - Personalization
  

2. Data pipeline
  - sufficiency: X->y
  - storage/access: [cloud, local premise, user device]
  - data prep, repr, posp: [label quality, missing values, multi-modal prep, feature-engineering & feature-selection, imbalance classes, ]
  - privacy, bias

3. Model training
  - baseline modeling: [random baseline, human baseline, simple model baseline]
  - hyperparameter tuning [grid-search, random, bayesian, ...]
  - model selection: interpret vs performance
  - scaling 
    - data paralellism: chop up data
    - model paralellism: chop up model (eg. seqential/paralell layers by layers)
    - reduce memory footprint by reducing floating point precision (32 bits -> 16 bits)=> faster computation, bigger batch size
  - debugging
    - start simple & gradually add more components VS. plud & play existing repo
    - overfit a single batch: loss decrease as epoch increase
    - set a random seed to make your model training reproducible
  
```
There are many reasons that can cause a model to perform poorly:

Theoretical constraints: e.g. wrong assumptions, poor model/data fit.
Poor model implementation: the more components a model has, the more things that can go wrong, and the harder it is to figure out which goes wrong.
Snobby training techniques: e.g. call model.train() instead of model.eval() during evaluation.
Poor choice of hyperparameters: with the same implementation, a set of hyperparameters can give you the state-of-the-art result but another set of hyperparameters might never converge.
Data problems: mismatched inputs/labels, over-preprocessed data, noisy data, etc.
```

4. Model serving
- to satesfy constraints on [user, business, ML sys]
- user feedback

References
1. Huyen Chip: [Design a machine learning system](https://huyenchip.com/machine-learning-systems-design/design-a-machine-learning-system.html#design-a-machine-learning-system-dwGQI5R)

1. 


# nbdev_colab

### Why
- to develop your package, to test code, to write documentation from 1 source of truth: your colab notebooks
- no beefy local computer needed, everything is computed with a colab notebook and stored on your Google drive

### How

Step0: clone this template to your github account.

Step1: create a new repo (e.g. name it 'my_amazing_project') using this template. 

Step2: to clone your newly created repo in your Google drive, use colab to open this notebook [git_clone_my_amazing_project_to_gdrive.ipynb](https://github.com/wjlgatech/nbdev_colab/blob/master/git_clone_my_amazing_project_to_gdrive.ipynb) and run it through.

Step3: to learn what to do next, start at notebook `nb/00_core.ipynb` from your Google Drive


Known bugs that need to be sorted (any help is welcome):

* ReadMe not updating after updating `index.ipynb`
* Tests are not passing


