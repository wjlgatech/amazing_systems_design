# ML systems Design

## 4-Component Design

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

## Important concepts

- [Reference](https://www.linkedin.com/feed/update/urn:li:activity:7027637200456429569/)
System design plays a crucial role in the success of any machine learning project. Based on my experience I have tried to compile top 10 system design concepts necessary for successful ML project implementation:

✅𝐃𝐚𝐭𝐚 𝐌𝐚𝐧𝐚𝐠𝐞𝐦𝐞𝐧𝐭: A system for collecting, cleaning, storing and transforming data to prepare it for use in ML models. E.g.: A data pipeline for collecting and cleaning customer data from various sources for use in a customer segmentation model.

✅𝐌𝐨𝐝𝐞𝐥 𝐒𝐞𝐫𝐯𝐢𝐧𝐠: A system for deploying ML models in a production environment. E.g.: A model serving system for deploying a customer segmentation model to make predictions in real-time.

✅𝐌𝐨𝐝𝐞𝐥 𝐌𝐚𝐧𝐚𝐠𝐞𝐦𝐞𝐧𝐭: A system for tracking the performance and accuracy of ML models over time. E.g.: A model management system for tracking the performance and accuracy of a customer segmentation model and selecting the best model for deployment.

✅𝐒𝐜𝐚𝐥𝐚𝐛𝐢𝐥𝐢𝐭𝐲: The ability of a system to handle increasing amounts of data and traffic. E.g.: Scaling a machine learning system to handle millions of customer data points and thousands of predictions per second.

✅𝐇𝐢𝐠𝐡 𝐀𝐯𝐚𝐢𝐥𝐚𝐛𝐢𝐥𝐢𝐭𝐲: Designing systems that can provide continuous service even in the face of failures. E.g.: A machine learning system that can continue to make predictions even if one of its servers goes down.

✅𝐋𝐨𝐚𝐝 𝐁𝐚𝐥𝐚𝐧𝐜𝐢𝐧𝐠: Distributing workloads evenly across multiple servers to ensure efficient use of resources. E.g.: Balancing the load of incoming predictions across multiple servers to prevent any single server from being overwhelmed.

✅𝐂𝐚𝐜𝐡𝐢𝐧𝐠: Storing frequently accessed data in memory to reduce the load on the backend. E.g.: Caching frequently requested customer data in memory to reduce the load on the database.

✅𝐌𝐨𝐧𝐢𝐭𝐨𝐫𝐢𝐧𝐠 𝐚𝐧𝐝 𝐋𝐨𝐠𝐠𝐢𝐧𝐠: Collecting and analyzing data about system performance to detect and resolve issues. E.g.: Monitoring and logging performance metrics of a ML system to detect and resolve any performance bottlenecks.

✅𝐂𝐨𝐧𝐭𝐢𝐧𝐮𝐨𝐮𝐬 𝐈𝐧𝐭𝐞𝐠𝐫𝐚𝐭𝐢𝐨𝐧 𝐚𝐧𝐝 𝐃𝐞𝐩𝐥𝐨𝐲𝐦𝐞𝐧𝐭 (𝐂𝐈/𝐂𝐃): Automating the process of building, testing, and deploying ML models. E.g.: An automated CI/CD process for building, testing, and deploying a customer segmentation model.

✅𝐄𝐱𝐩𝐥𝐚𝐢𝐧𝐚𝐛𝐢𝐥𝐢𝐭𝐲: Making the predictions of machine learning models transparent and interpretable to stakeholders. E.g.: An explainability system for providing clear and concise explanations of the predictions made by a customer segmentation model.

By incorporating these concepts into your ML system design, you'll be well on your way to building high-performing, reliable, and interpretable machine learning systems.


## Software Systems Design by Alex Xu
From 0 to Millions: A Guide to Scaling Your App 
- [1 Single Server](https://blog.bytebytego.com/p/from-0-to-millions-a-guide-to-scaling?utm_source=substack&utm_medium=email), 
- [2 Cashe, DB sharding](https://blog.bytebytego.com/p/from-0-to-millions-a-guide-to-scaling?utm_source=substack&utm_medium=email), 
- [3 frontend: Single-Page-Application, backend: Serverless](https://blog.bytebytego.com/p/from-0-to-millions-a-guide-to-scaling-b53?utm_source=substack&utm_medium=email), 
- [4 Read Replicas, Caching, DB sharding](https://blog.bytebytego.com/p/from-0-to-millions-a-guide-to-scaling-47a?utm_source=substack&utm_medium=email),  
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


