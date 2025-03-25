# ALPINE

**ALPINE: Adaptive Layering of Phenotypic and Integrative Noise Extraction**

`ALPINE` is a semi-supervised non-negative matrix factorization (NMF) framework designed to effectively distinguish between multiple phenotypic conditions based on shared biological factors, while also providing direct interpretability of condition-associated genes. The entire package is developed in Python and supports GPU usage, significantly enhancing computational speed.

ALPINE can be useful for:

- Identifying condition-associated genes and cells.
- Studying the biological functions of condition-related genes.
- Removing batch effects from the data.

## Installation

Currently, `ALPINE` is not yet available on the PyPI repository. However, users can clone the entire repository and install the package in their environment by running:

```shell
# in the alpine folder
pip install -e .
```

**Important:**
ALPINE is implemented in [PyTorch](https://pytorch.org/), so users will need to install PyTorch separately to ensure compatibility with their specific CUDA or CPU version.

## Usage

The input data type for `ALPINE` is the `AnnData` format. Users should note that ALPINE's model is based on an NMF structure, which supports only **non-negative** values. ALPINE can be trained using either the entire gene list or a selection of highly variable genes.

ALPINE consists of two primary components: optimization and training. The associated classes can be imported as follows:

```python
from alpine import ALPINE, ComponentOptimizer
```

### 1. Optimization

ALPINE integrates a Bayesian optimizer to efficiently search for all necessary hyperparameters, allowing users to easily apply the optimized parameters directly within ALPINE.

```python
from alpine import ComponentOptimizer

# create optimization object with data and covariate keys
co = ComponentOptimizer(adata, covariate_keys=["cov_1", "cov_2"])

# start searching with given parameter range
params = co.bayesian_search(
    n_total_components_range=(50, 100), 
    alpha_W_range=(0, 1),
    orth_W_range=(0, 0.5),
    l1_ratio_range=(0, 1),
)
```

- `covariate_keys` specifies the categorical columns in `adata.obs` that will be used as covariates.
- `n_total_components_range` sets the range for the total number of components, including `n_components` for unguided embeddings and `n_covariate_components` for guided embeddings.
- `lam_power_range` defines the range for lambda values, spanning from \(10^1\) to \(10^5\).
- `orth_W_range`: The range for the orthogonal weight regularization on the \( W \) matrix, designed to encourage gene signatures to capture distinct patterns.
- `l1_ratio_range`: The range for the L1 ratio, controlling the balance between L1 (LASSO) and L2 (ridge) regularization.
- `alpha_W_range`: The range for the regularization weight on the \( W \) matrix, determining the influence of LASSO and ridge regularization on \( W \).


The `ComponentOptimizer` class offers a range of convenient and practical functions to help users monitor and extend their training process. See the analysis section below for more details.

### 2. Multi-condition disentangle using ALPINE

With ALPINE, you have the flexibility to either manually define the parameters you want to use or apply the optimized parameters learned from previous steps.

#### a. Training the model
1. **Manually specified parameters:**

```python
# user can maually specify desired paramteres
alpine_model = ALPINE(
    n_components = 30,
    n_covariate_components = [5, 5] 
    alpha_W = 0,
    lam = [1e+3, 1e+3],
    gpu = True
)
alpine_model.fit(adata, covariate_keys=["cov_1", "cov_2"])
```

2.  **Using optimized parameters from `ComponentOptimizer` (Recommend):**

```python
# if you use the ComponentOptimizer, you can simply plugin the parameteres learned from the last step
alpine_model = ALPINE(**param)
alpine_model.fit(adata, covariate_keys=["cov_1", "cov_2"])
```

Finally, users can save the trained embeddings directly into `adata` by running:

```python
alpine_model.store_embeddings(adata)

# the H embedding can be retrieved by
alpine_model.obsm["ALPINE_embedding"] # unguided embedding
alpine_model.obsm["cov_1"] # covariate embedding
alpine_model.obsm["cov_2"] # covariate embedding

# the W embedding
alpine_model.varm["ALPINE_embedding"] # unguided gene signature embedding
alpine_model.varm["cov_1"] # covariate gene signature embedding
alpine_model.varm["cov_2"] # covariate gene signature embedding

```
#### b. Get the decomposed matrices and counts

In addition to obtaining embeddings from `adata`, users can also retrieve the decomposed matrices from the `ALPINE` model by using:

```python
Ws, Hs, Bs = alpine_model.get_decomposed_matrices()
```
The order of the matrices in `Ws` and `Hs` follows the sequence of the covariate keys, with the unguided embeddings placed at the end. In contrast, the `Bs` matrices do not include the unguided portion; their order strictly adheres to the covariate keys.

To obtain the normalized counts that are free from batch effects and conditions, where:

```python
alpine_model.get_normalized_expression(adata)

# the normalized counts is in here
adata.obsm["normalized_expression"]
```

There are additional applications for our model; please refer to the next section for more details.

## More usage, and analysis

All analyses from the papers and case studies are stored in the ALPINE-analysis repository, where you can access a variety of resources. Additionally, the repository provides valuable tips for tuning the model.

- [ALPINE-anlaysis repo](https://github.com/ylaboratory/ALPINE-analysis)
- Save, load, and extend the optimization process.
- Retrieve condition-associated gene scores.
- Transform new, unseen data using the trained model.


## Citation
