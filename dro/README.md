# Comparing Fair ML algorithms

This project aims to compare various machine learning algorithms 
that aim to improve the fairness of the machine learning.

# Algorithms
The table below shows each of the algorithms, their implementation source, the original paper(s) they are from 
or based on, and whether they use demographic information directly.

| Algorithm                  | Available | Uses Demographics | Hyperparameters | Implementation Source     | Paper                                                                                     | Docs | Code                                                                                                             |
|----------------------------|-----------|-------------------|-----------------|---------------------------|-------------------------------------------------------------------------------------------|--|------------------------------------------------------------------------------------------------------------------|
| AdversarialDebiasing       | Yes       | Yes               | Yes             | AIF360                    | [ACM](https://dl.acm.org/doi/10.1145/3278721.3278779)                                                                                      | [Docs](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.AdversarialDebiasing.html) | AIF360                                                                                                           |
| VariationalFairAutoEncoder | No        | Yes               | Yes             | N/A                       | [PapersWithCode](https://paperswithcode.com/paper/the-variational-fair-autoencoder)       | N/A | [GitHub](https://github.com/NCTUMLlab/Huang-Ching-Wei)                                                           |
| MetaFairClassifier         | Yes       | Yes               | Yes             | AIF360                    | [ArXiv](https://arxiv.org/abs/1806.06055)                                                                                      | [Docs](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.MetaFairClassifier.html) | AIF360                                                                                                           |
| PrejudiceRemover           | Broken    | Yes               | Yes             | AIF360                    | [Springer](https://link.springer.com/chapter/10.1007/978-3-642-33486-3_3)                                                                                      | [Docs](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.PrejudiceRemover.html) | AIF360                                                                                                           |
| GridSearchReduction        | Broken    | Yes               | Yes             | AIF360                    | [ArXiv 1](https://arxiv.org/abs/1803.02453) [ArXiv 2](https://arxiv.org/abs/1905.12843)                                                                                      | [Docs](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.GridSearchReduction.html) | AIF360                                                                                                           |
| AdversarialReweighting     | Yes       | No                | Yes             | Google Research (Adapted) | [AlexBeutel](http://alexbeutel.com/papers/NeurIPS-2020-fairness-without-demographics.pdf) | N/A | [Google Research GitHub](https://github.com/google-research/google-research/tree/master/group_agnostic_fairness) |
| GerryFairClassifier        | Yes       | No                | Yes             | AIF360                    | [PMLR](https://proceedings.mlr.press/v80/kearns18a.html) [NSF](https://par.nsf.gov/servlets/purl/10100406)                                                                                      | [Docs](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.GerryFairClassifier.html) | AIF360                                                                                                           |
| RepeatedLossMinimization   | No        | No                | Yes             | N/A                       | [ArXiv](https://arxiv.org/abs/1806.08010)                                                 | N/A | [Codalab](https://worksheets.codalab.org/worksheets/0x17a501d37bbe49279b0c70ae10813f4c)                          |                                                |

# How to use this project
Create a Python environment. 
The code was tested with Python 3.7.13


Install required packages:
```
pip install -r requirements.txt
```


Run the code:
The code can be run from the command line, 
the interface is powered by [Hydra](https://hydra.cc/).
The code for single model can be run using:
```
python main.py
```

Hydra uses the config file at `/conf/config.yaml` to run the code.
The `model_params` structure in the yaml can be used to change the 
hyperparameters of the model; names must match exactly.

Hydra also allows automatically running multiple versions of the configs (in series, plugins are available for parallel runs)
A command using multirun for different number of epochs (5, 25, and 50) would be:
```
python main.py model_params.num_epochs=5,25,50 --multirun
```
Refer to [Hydra documentation](https://hydra.cc/docs/latest/user/config.html) for more information.

## Config file Details
The config is set on a .yaml file at `/conf/config.yaml`.

The dataset to be used is set on the `dataset_name` field, currently supports: 
  - `'uci_adult'`
  - `'compas'`

The model to be used is set on the `model_name` field, currently supports: 
  - `'AdversarialDebiasing'`
  - `'VariationalFairAutoEncoder'`
  - `'MetaFairClassifier'`
  - `'PrejudiceRemover'`
  - `'GridSearchReduction'`
  - `'AdversarialReweighting'`
  - `'GerryFairClassifier'`
  - `'RepeatedLossMinimization'`

# Known issues & TODOs
- The 'PrejudiceRemover' model is not working properly, throws an error when trying to evaluate ("Empty input file"), might be an issue with the hydra config?
- The 'Variational_Fair_AutoEncoder' model is not implemented yet
- The 'GridSearchReduction' model needs an estimator to be given, this is not implemented yet
- For 'AdversarialReweighting', only supporting UCI Adult dataset right now. Need to create `.json` (for `mean_std.json`, `vocabulary.json`, `IPS_example_weights_with_label.json`, and `IPS_example_weights_without_label.json`) files for other datasets (see `group_agnostic_fairness/README.md`)

# Changelog
May 6th, 2022:
- Added Hydra for config and cli
- Added README
- Tested AIF360 algorithm usage
- Tested group agnostic fairness adapter usage
- Added code for RepeatedLossMinimization