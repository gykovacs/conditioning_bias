# Implementation of the study "The Conditioning Bias in Binary Decision Trees and Random Forests and Its Elimination"

## Introduction

In this repository, we share the all the implementations required for the complete reproduction of the results presented in the study.

Citation:

```bibtex
@article{conditioning-bias,
    author = {G\'abor T\'im\'ar and Gy\"orgy Kov\'acs},
    title = {The Conditioning Bias in Binary Decision Trees and Random Forests and Its Elimination},
    year = {2023}
}
```

## Contents

The contents of the repository:

* `conditioning_bias`: the Python package containing the implementations of the proposed methods;
* `study`: the notebooks that can be used to reproduce the results of the study.

## Reproducing the results

In this section, we provide a detailed description of how to reproduce the study.

### Prerequisites

First, clone the repository:

```bash
> git clone
```

Create a new Python virtual environment (we use the `conda` environment manager):

```bash
> conda create -n conditioning-bias python==3.11 jupyter ipykernel
```

Activate the environment:

```bash
> conda activate conditioning-bias
```

Enter the repository and install the package:

```bash
> cd conditioning-bias
> pip install .
```

### Executing the steps of the analysis

The repository contains the all .csv data files containing the results of the individual steps. Based on these files, any step can be rerun individually.

#### Jupyter

The main steps of the study are arranged as individual Jupyter notebooks. In order to execute them, fire up Jupyter or any IDE supporting the notebook, and activate the `conditioning-bias` kernel to be used.

#### Configuration

The entire experiment can be configured by overwriting the variables in the `study/config.py` module.

#### Datasets

The datasets are loaded and filtered through the tools of the `common-datasets` package.  The filtering parameters can be adjusted in the `study/config.py` module. For the ease of use, some preprocessing is applied in the `study/datasets.py` module.

#### Step 0 - The decision tree used for illustration

The decision tree used for illustration in the paper is constructed in the notebook `study/000-illustration-decision-tree.ipynb`

#### Step 1 - Model selection

#### Step 2 - Generating the summary of the datasets

#### Step 3 - Checking the attributes of the toy datasets in `sklearn`

#### Step 4 - Generating the statistics of nodes with thresholds on domain values

#### Step 5 - The cross-validated evaluation of all variations of the estimators

#### Step 6 - The distributions of AUC and r2 scores

#### Step 7 - The analysis of the results regarding the existence of the bias

#### Step 8 - The analysis of the results regarding the performance of the proposed method

