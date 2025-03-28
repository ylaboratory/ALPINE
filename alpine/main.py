import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch

from kneed import KneeLocator
from tqdm import tqdm
from typing import Optional, Union, List, Tuple
from copy import deepcopy
from sklearn.utils.class_weight import compute_sample_weight
from contextlib import nullcontext


class ALPINE:
    """
    ALPINE: A class for performing matrix factorization with covariate components.
    This class implements a matrix factorization model that incorporates covariate components 
    to decompose input data into low-dimensional representations. It supports non-negative 
    matrix factorization with optional orthogonality constraints and loss functions such as 
    KL-divergence.
    Attributes:
        n_components (int): Number of components for the main factorization.
        n_covariate_components (List[int]): List of integers specifying the number of components 
            for each covariate.
        alpha_W (float): Regularization parameter for the W matrix.
        orth_W (float): Orthogonality constraint weight for the W matrix.
        l1_ratio (float): Ratio of L1 regularization in the W matrix.
        random_state (Optional[int]): Random seed for reproducibility.
        gpu (bool): Whether to use GPU for computations.
        device (torch.device): The device (CPU or GPU) used for computations.
        loss_type (str): Loss function type ('kl-divergence' or 'frobenius').
        eps (float): Small constant to avoid division by zero.
        n_all_components (List[int]): List of all components (covariate + main).
        total_components (int): Total number of components.
        lam (List[float]): Regularization weights for covariate components.
        scale_needed (bool): Whether to scale the decomposed matrices.
        loss_history (pd.DataFrame): History of loss values during training.
    Methods:
        fit(X, covariate_keys, max_iter=None, batch_size=None, sample_weights=True, verbose=True, **kwargs):
            Fits the model to the input data and covariates.
        transform(X, n_iter=None, lam=None, use_label=True):
            Transforms new data using the fitted model.
        get_decomposed_matrices():
            Returns the decomposed matrices (Ws, Hs, Bs).
        get_W_and_H(idx=-1):
            Returns the W and H matrices for a specific component.
        W_concat(idx):
            Concatenates W matrices for the specified indices.
        H_concat(idx):
            Concatenates H matrices for the specified indices.
        get_conditional_gene_scores():
            Computes conditional gene scores for each covariate.
        get_reconstructed_counts(idx=[-1]):
            Reconstructs the input data using the specified components.
        get_normalized_expression(adata, library_size=1e+4):
            Computes normalized expression values and stores them in the AnnData object.
        compute_sig_assoc(covariate_key, sample_labels):
            Computes significant associations between covariates and sample labels.
        store_embeddings(adata, embedding_name=None):
            Stores embeddings in the AnnData object.
        _check_params():
            Validates the input parameters.
        _check_input():
            Validates the input data and covariates.
        _check_decomposed_matrices():
            Initializes or validates the decomposed matrices.
        _check_non_negative(X):
            Checks if the input matrix is non-negative.
        _process_lam(lam):
            Processes the regularization parameter for covariates.
        _process_input(X, y):
            Processes the input data and covariates.
        _to_dummies(y):
            Converts categorical covariates to dummy variables.
        _to_dummies_from_train(i, y):
            Converts new covariates to dummy variables based on training data.
        _kl_divergence(y, y_hat):
            Computes the KL-divergence loss.
        _get_best_max_iter(loss_history):
            Determines the optimal number of iterations using the elbow method.
        _fit(sample_weights, verbose):
            Performs the matrix factorization.
        _transform_wo_labels(X, n_iter=None, batch_size=None, lam=None):
            Transforms new data without using covariate labels.
        _transform(X, n_iter=None, batch_size=None, lam=None):
            Transforms new data using covariate labels.
    """
    
    def __init__(
        self,
        n_covariate_components: Union[List[int], int] ,
        n_components: int = 30,
        alpha_W = 0,
        orth_W = 0, 
        l1_ratio = 0,
        lam: Optional[Union[List[float], List[int], float, int]] = None,
        gpu: bool = True,
        scale_needed: bool = True,
        loss_type = 'kl-divergence',
        random_state: Optional[int] = 42,
        eps: float = 1e-10,
        **kwargs
    ):
        
        self.n_components = n_components
        self.n_covariate_components = [n_covariate_components] if isinstance(n_covariate_components, int) else n_covariate_components
        self.alpha_W = alpha_W
        self.orth_W= orth_W
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.gpu = gpu
        self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
        self.loss_type = loss_type
        self.eps = eps
        self.n_all_components = self.n_covariate_components + [self.n_components]
        self.total_components = sum(self.n_covariate_components) + self.n_components
        self.lam = self._process_lam(lam)
        self.scale_needed = scale_needed
        self._check_params()


    def fit(
            self, X: ad.AnnData, 
            covariate_keys: Union[List[str], str], 
            max_iter: Optional[int] = None,
            batch_size: Optional[int] = None, 
            sample_weights: bool = True,
            verbose: bool = True,
            **kwargs
        ) -> None:
        """
        Fits the model to the given data and covariates.
        Parameters:
            X (ad.AnnData): The input data in AnnData format.
            covariate_keys (Union[List[str], str]): Keys for covariates in the input data.
            max_iter (Optional[int], optional): Maximum number of iterations for fitting. If not provided, it will be determined automatically. Defaults to None.
            batch_size (Optional[int], optional): Batch size for processing. If not provided, it defaults to one-third of the number of features in X. Defaults to None.
            sample_weights (bool, optional): Whether to use sample weights during fitting. Defaults to True.
            verbose (bool, optional): Whether to display verbose output during fitting. Defaults to True.
            **kwargs: Additional keyword arguments for specifying precomputed matrices:
                - Ws: Precomputed W matrices.
                - Hs: Precomputed H matrices.
                - Bs: Precomputed B matrices.
        Returns:
            None: The function modifies the instance attributes in place.
        Notes:
            - If `max_iter` is not provided, the function will automatically determine the best value by performing an initial fit and analyzing the loss history.
            - The function processes the input data and covariates, checks their validity, and initializes or validates decomposed matrices.
            - If scaling is required, the function scales the W, H, and B matrices appropriately.
            - The fitted matrices (W, H, B) are saved as instance attributes for further use.
        """
        self.max_iter = max_iter
        self.X, self.y_input, self.condition_names = self._process_input(X, covariate_keys)
        self.y, self.y_labels, self.M_y = zip(*[self._to_dummies(yi) for yi in self.y_input])
        self.batch_size = batch_size if batch_size is not None else int(self.X.shape[1] / 3)
        self._check_input()
        
        self.Ws = kwargs.get('Ws', None)
        self.Hs = kwargs.get('Hs', None)
        self.Bs = kwargs.get('Bs', None)
        self._check_decomposed_matrices()

        # Automatically search for the best max_iter if not provided
        if max_iter is None:
            # for the scanning purpose
            self.max_iter = 200
            _, _, _ = self._fit(sample_weights, verbose=False)
            self.max_iter = self._get_best_max_iter(self.loss_history)
            
            if self.max_iter == 0 or self.max_iter is None:
                print("The max_iter detection is unable to find a good max_iter. The max_iter set to 200 as default.")
                self.max_iter = 200
        
        Ws, Hs, Bs = self._fit(sample_weights, verbose)

        if self.scale_needed:
            # with np.errstate(divide='ignore', invalid='ignore'):
            #     Bs = [_b / np.sum(_b, axis=0) for _b in Bs]
            #     Bs = [np.where(np.isnan(_b), 0, _b) for _b in Bs]

            for i in range(len(Ws)):
                w_scaler = Ws[i].sum(axis=0)
                Ws[i] /= w_scaler
                Hs[i] *= w_scaler[:, np.newaxis]

                if i < len(self.n_covariate_components):
                    Bs[i] /= w_scaler

        # Save the matrices
        self.Ws, self.Hs, self.Bs = Ws, Hs, Bs
        self.W = np.hstack(self.Ws)
        self.H = np.vstack(self.Hs)
        

    def transform(
            self,
            X: ad.AnnData,
            n_iter: Optional[int] = None,
            lam = None,
            use_label = True,
        ) -> List[np.ndarray]:
        """
        Transforms the input AnnData object using the specified parameters.

        Parameters:
            X (ad.AnnData): The input AnnData object to be transformed.
            n_iter (Optional[int], optional): The number of iterations to perform. 
                If None, defaults to 1000. If `use_label` is True and `n_iter` is None, 
                an optimized number of iterations will be determined. Defaults to None.
            lam (optional): A parameter used during the transformation. The specific 
                purpose depends on the implementation of `_transform`.
            use_label (bool, optional): If True, the transformation will use labels 
                during processing. If False, the transformation will be performed 
                without labels. Defaults to True.

        Returns:
            List[np.ndarray]: A list of numpy arrays resulting from the transformation.
        """
        if n_iter is None:
            n_iter = 1000

        # if n_iter is None, will find a optimized n_iter
        if use_label:
            Hs, _ = self._transform(X, n_iter=n_iter, lam=lam)
        else:
            Hs, _ = self._transform_wo_labels(X, n_iter=n_iter)
        return Hs


    def get_decomposed_matrices(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Retrieve the decomposed matrices (Ws, Hs, Bs) from the model.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
                - A deep copy of the list of W matrices.
                - A deep copy of the list of H matrices.
                - A deep copy of the list of B matrices.

        Raises:
            RuntimeError: If the model has not been fitted yet. Ensure that the `fit()` 
                          method is called before invoking this method.
        """
        if not hasattr(self, 'Ws') or not hasattr(self, 'Hs') or not hasattr(self, 'Bs'):
            raise RuntimeError("Model has not been fitted. Call fit() before get_decomposed_matrices().")
        return deepcopy(self.Ws), deepcopy(self.Hs), deepcopy(self.Bs)
    

    def get_W_and_H(self, idx: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the W and H matrices from the model.

        Parameters:
            idx (int): The index of the matrices to retrieve. Defaults to -1, 
                       which retrieves the last set of matrices.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the W and H matrices.

        Raises:
            RuntimeError: If the model has not been fitted and the W and H matrices 
                          are not available.
        """
        if not hasattr(self, 'W') or not hasattr(self, 'H'):
            raise RuntimeError("Model has not been fitted. Call fit() before get_W_and_H().")
        return deepcopy(self.Ws[idx]), deepcopy(self.Hs[idx])

    
    def W_concat (self, idx: List[int]):
        return np.concatenate([self.Ws[i] for i in idx], axis=1)


    def H_concat (self, idx: List[int]):
        return np.concatenate([self.Hs[i] for i in idx], axis=0)


    def get_conditional_gene_scores (self):
        """
        Calculate conditional gene scores for each set of matrices in the model.

        This method computes the conditional gene scores by performing matrix 
        multiplications and normalizations for each set of matrices (Ws, Hs, and y) 
        in the model. The resulting scores are stored in a list of pandas DataFrames.

        Returns:
            List[pd.DataFrame]: A list of DataFrames where each DataFrame contains 
            the conditional gene scores for a corresponding set of matrices. The 
            columns of each DataFrame are labeled using the corresponding y_labels.
        """
        conditional_gene_scores = []
        for i in range(len(self.Bs)):
            gene_scores = (self.Ws[i] @ self.Hs[i] @ self.y[i].T) / self.y[i].sum(axis=1)
            conditional_gene_scores.append(pd.DataFrame(gene_scores, columns=self.y_labels[i]))
        return conditional_gene_scores


    def get_reconstructed_counts(self, idx:Optional[List[int]] = [-1]):
        """
        Compute the reconstructed counts by summing the matrix products of Ws and Hs 
        for the specified indices.

        Args:
            idx (Optional[List[int]]): A list of indices specifying which matrices 
                to include in the reconstruction. Defaults to [-1], which typically 
                indicates the last element or a specific convention in the context 
                of the implementation.

        Returns:
            np.ndarray: The reconstructed counts as a 2D NumPy array obtained by 
            summing the matrix products of Ws and Hs for the specified indices.
        """
        return np.sum([self.Ws[i] @ self.Hs[i] for i in idx], axis=0)

    
    def get_normalized_expression(self, adata, library_size=1e+4):
        """
        Normalize the expression data using a specified library size.

        This method computes the normalized expression matrix from the 
        ALPINE embedding and stores it in the `obsm` attribute of the 
        AnnData object.

        Parameters:
            adata (AnnData): The annotated data matrix containing the 
                ALPINE embedding in `obsm["ALPINE_embedding"]`.
            library_size (float, optional): The target sum for normalization. 
                Default is 1e+4.

        Returns:
            None: The normalized expression matrix is stored in 
            `adata.obsm["normalized_expression"]`.
        """
        expression =  adata.obsm["ALPINE_embedding"] @ self.Ws[-1].T
        exp = ad.AnnData(expression)
        sc.pp.normalize_total(exp, target_sum=library_size)
        adata.obsm["normalized_expression"] = exp.X
    
    def compute_sig_assoc(self, covariate_key: str, sample_labels: pd.Series):
        """
        Compute significant associations between a covariate and sample labels.
        This method calculates the significant associations for a given covariate
        key by projecting the sample labels onto the corresponding matrix `H` 
        associated with the covariate in the model.
        Args:
            covariate_key (str): The key identifying the covariate in the model.
                                 Must exist in `self.condition_names`.
            sample_labels (pd.Series): A pandas Series containing the sample labels
                                       to be analyzed. These labels will be one-hot
                                       encoded.
        Returns:
            pd.DataFrame: A DataFrame containing the significant associations, where
                          rows correspond to the components of the covariate and 
                          columns correspond to the unique sample labels.
        Raises:
            ValueError: If the `covariate_key` does not exist in `self.condition_names`.
            ValueError: If the number of samples in `sample_labels` does not match
                        the number of samples in the model.
        """
        
        if covariate_key not in self.condition_names:
            raise ValueError("The covariate key does not exist in the model.")
        idx = self.condition_names.index(covariate_key)

        y_df = pd.get_dummies(sample_labels, dtype=float)
        y = y_df.values

        H = self.Hs[idx]
        if H.shape[1] != y.shape[0]:
            raise ValueError("The number of samples in the input data does not match the number of samples in the model.")
        
        sig_assoc = (H @ y) / y.sum(axis=0)
        sig_assoc_df = pd.DataFrame(sig_assoc, columns=y_df.columns)
        return sig_assoc_df

    def store_embeddings(self, adata: ad.AnnData, embedding_name: Optional[Union[List[str], str]] = None):
        """
        Stores the embeddings from the decomposition into the provided AnnData object.

        Parameters:
        -----------
        adata : ad.AnnData
            The AnnData object where the embeddings will be stored.
        embedding_name : Optional[Union[List[str], str]], default=None
            The names of the embeddings to be stored. If a single string is provided, it will be
            converted to a list. The length of `embedding_name` must match the number of decomposed
            matrices (`self.n_all_components`). If not provided, the method will use `self.condition_names`
            as the embedding names.

        Raises:
        -------
        ValueError
            If `embedding_name` is provided but its length does not match the number of decomposed matrices.
        ValueError
            If `embedding_name` is not provided and `self.condition_names` is not defined.

        Notes:
        ------
        - The embeddings are stored in the `obsm` and `varm` attributes of the AnnData object.
        - The last embedding is always stored with the key "ALPINE_embedding" in both `obsm` and `varm`.
        """
        if embedding_name is not None:
            if isinstance(embedding_name, str):
                embedding_name = [embedding_name]
            if len(embedding_name) != len(self.n_all_components):
                raise ValueError("embedding_name must have the same length as all decomposed matrices!")
            for i, name in enumerate(embedding_name):
                adata.obsm[name] = self.Hs[i].T
                adata.varm[name] = self.Ws[i]
        else:
            if not hasattr(self, 'condition_names'):
                raise ValueError("Please provide the embedding names for all decomposed matrices!")

            for i, name in enumerate(self.condition_names):
                adata.obsm[name] = self.Hs[i].T
                adata.varm[name] = self.Ws[i]

        adata.obsm["ALPINE_embedding"] = self.Hs[-1].T
        adata.varm["ALPINE_embedding"] = self.Ws[-1]

    
    def _check_params(self):
        if not isinstance(self.n_components, int) or self.n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        
        if not all(isinstance(n, int) and n > 0 for n in self.n_all_components):
            raise ValueError("n_batch_components must be a list of positive integers")
        
        if self.lam is None:
            pass
        else:
            if len(self.lam) != len(self.n_covariate_components):
                raise ValueError("lam must have the same length as n_all_components")
        
            if not all(isinstance(lam, (float, int)) and lam >= 0 for lam in self.lam):
                raise ValueError("lam must be a list of non-negative floats or integers")

        if self.random_state is not None and not isinstance(self.random_state, int):
            raise ValueError("random_state must be None or an integer")


    def _check_input(self):
        if not isinstance(self.X, np.ndarray) or self.X.ndim != 2:
            raise ValueError("X must be a 2D numpy array.")
        if not all(isinstance(yi, np.ndarray) and yi.ndim == 2 for yi in self.y):
            raise ValueError("All elements of y must be 2D numpy arrays.")
        if len(self.y) != len(self.n_covariate_components):
            print(len(self.y), len(self.n_covariate_components))
            raise ValueError("Number of batch components must match the number of y matrices.")
        if any(yi.shape[1] != self.X.shape[1] for yi in self.y):
            raise ValueError("All y matrices must have the same number of columns as X.")


    def _check_decomposed_matrices(self):

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)
                
        n_feature, n_sample = self.X.shape

        if self.Ws is not None:
            self.Ws = [torch.tensor(w, dtype=torch.float32, device=self.device) for w in self.Ws]
        else:
            self.Ws = [torch.rand((n_feature, k), dtype=torch.float32, device=self.device) for k in self.n_all_components]
            
        if self.Hs is not None:
            self.Hs = [torch.tensor(h, dtype=torch.float32, device=self.device) for h in self.Hs]
        else:
            self.Hs = [torch.rand((k, n_sample), dtype=torch.float32, device=self.device) for k in self.n_all_components]

        if self.Bs is not None:
            self.Bs = [torch.tensor(b, dtype=torch.float32, device=self.device) for b in self.Bs]
        else:
            self.Bs = [torch.rand((_y.shape[0], k), dtype=torch.float32, device=self.device) for (_y, k) in zip(self.y, self.n_covariate_components)]

        self.M_y = [torch.tensor(m, dtype=torch.float32, device=self.device) for m in self.M_y]

    def _check_non_negative(self, X):
        if not np.all(X >= 0):
            return False


    def _process_lam(self, lam):
        if lam is None:
            return None
        if isinstance(lam, (float, int)):
            return [lam] * len(self.n_covariate_components)
        if len(lam) != len(self.n_covariate_components):
            raise ValueError("Length of lam must match n_batch_components")
        return lam


    def _process_input(self, X: ad.AnnData, y: Union[List[str], str]):
        if isinstance(X, ad.AnnData):
            if isinstance(X.X, (np.ndarray, np.generic)):
                X_processed = X.X.T
            else:
                X_processed = X.X.toarray().T
            condition_names = [y] if isinstance(y, str) else y
            y_processed = [X.obs[col] for col in condition_names]
        else:
            raise ValueError("X must be either an AnnData object or a numpy array.")
        return X_processed, y_processed, condition_names
    

    def _to_dummies(self, y):
        dummies_df = pd.get_dummies(y, dtype=float)
        labels = dummies_df.columns.tolist()
        mask = np.ones_like(dummies_df.values, dtype=float)
        mask[dummies_df.sum(axis=1) == 0] = 0

        return dummies_df.values.T, labels, mask.T

    def _to_dummies_from_train (self, i, y):
        dummies_df = pd.get_dummies(y, dtype=float)
        dummies_df = dummies_df.reindex(columns=self.y_labels[i], fill_value=0)
        mask = np.ones_like(dummies_df.values, dtype=float)
        mask[dummies_df.sum(axis=1) == 0] = 0

        return dummies_df.values.T, mask.T
    

    def _kl_divergence(self, y, y_hat):
        if torch.is_tensor(y):
            return torch.sum(y * torch.log(torch.clamp(y / torch.clamp(y_hat, min=self.eps), min=self.eps)) - y + y_hat)
        else:
            return np.sum(y * np.log(np.clip(y / np.clip(y_hat, a_min=self.eps, a_max=None), a_min=self.eps, a_max=None)) - y + y_hat)


    def _get_best_max_iter (self, loss_history):
        kneedle = KneeLocator(np.arange(0, loss_history.shape[0]), 
                              np.log10(loss_history["reconstruction_loss"].values), 
                              curve='convex', direction='decreasing')
        return kneedle.elbow
    

    def _fit(self, sample_weights, verbose):

        X = torch.tensor(self.X, dtype=torch.float32, device=self.device)
        n_sample = X.shape[1]
        y = [torch.tensor(y, dtype=torch.float32, device=self.device) for y in self.y]

        Ws, Hs, Bs, M_y = deepcopy(self.Ws), deepcopy(self.Hs), deepcopy(self.Bs), deepcopy(self.M_y)

        if self.lam is None:
            W = torch.cat(Ws, dim=1)
            H = torch.cat(Hs, dim=0)
            recon_loss = torch.norm(X - W @ H, 'fro')**2
            if self.loss_type == "kl-divergence":
                label_loss = [self._kl_divergence(_y, _b @ _h) for _y, _b, _h in zip(y, Bs, Hs[:len(self.n_covariate_components)])]
                self.lam = [(recon_loss.item() / 1e+3) / ll.clamp(min=self.eps).item() for ll in label_loss]
            else:
                label_loss = [torch.norm(_y - _b @ _h, 'fro')**2 for _y, _b, _h in zip(y, Bs, Hs[:len(self.n_covariate_components)])]
                self.lam = [(recon_loss.item() / 1e+2) / ll.clamp(min=self.eps).item() for ll in label_loss]

        loss_history = []

        # orthogonal constraint
        orthogonal_matrix = torch.zeros((self.total_components, self.total_components), dtype=torch.float32, device=self.device)
        start_idx = 0
        for i in range(len(self.n_all_components)):
            end_idx = start_idx + Ws[i].shape[1]
            k = Ws[i].shape[1]
            weights = self.orth_W * (self.total_components / k)
            orthogonal_matrix[start_idx:end_idx, start_idx:end_idx] = weights * torch.ones(k, k, device=self.device) - torch.eye(k, device=self.device)
            start_idx = end_idx

        pbar = tqdm(total=self.max_iter, desc="Epoch", ncols=150) if verbose else nullcontext()
        with pbar:
            for _ in range(self.max_iter):
                for _ in range(0, n_sample, self.batch_size):
                    if isinstance(sample_weights, bool) and sample_weights:
                        joint_labels = ["+".join(str(y.iloc[j]) for y in self.y_input) for j in range(len(self.y_input[0]))]
                        sample_weights = compute_sample_weight(class_weight='balanced', y=joint_labels)
                        sample_weights = torch.tensor(sample_weights / np.sum(sample_weights), dtype=torch.float32, device=self.device)
                        indices = torch.multinomial(sample_weights, min(self.batch_size, len(sample_weights)), replacement=False)
                    else:
                        indices = torch.randperm(n_sample, device=self.device)[:self.batch_size]

                    X_sub = X[:, indices]
                    Hs_sub = [h[:, indices] for h in Hs]
                    y_sub = [_y[:, indices] for _y in y]
                    M_y_sub = [m[:, indices] for m in M_y]

                    # update W
                    W = torch.cat(Ws, dim=1)
                    H_sub = torch.cat(Hs_sub, dim=0)

                    W_numerator = X_sub @ H_sub.T
                    # W_denominator = W @ H_sub @ H_sub.T

                    # orthogonal constraint + l2 norm
                    W_denominator = W @ (H_sub @ H_sub.T +  (1 - self.l1_ratio) * self.alpha_W * torch.eye(self.total_components, device=self.device) + orthogonal_matrix)

                    # orthogonal constraint
                    # W_denominator = W @ (H_sub @ H_sub.T + orthogonal_matrix)

                    # l1 norm
                    W_denominator += self.l1_ratio * self.alpha_W * torch.ones_like(W_denominator)

                    # orthogonal on the entire W
                    # W_denominator = W @ (H_sub @ H_sub.T + self.alpha_W * (torch.ones_like(H_sub @ H_sub.T) - torch.eye(H_sub.shape[0], device=self.device)))



                    W *= torch.div(torch.clamp(W_numerator, min=self.eps), torch.clamp(W_denominator, min=self.eps))

                    start_idx = 0
                    for idx, w in enumerate(Ws):
                        end_idx = start_idx + w.shape[1]
                        Ws[idx] = W[:, start_idx:end_idx]
                        start_idx = end_idx

                    for b, (M, y_b, B_b, H_b) in enumerate(zip(M_y_sub, y_sub, Bs, Hs_sub[:len(self.n_covariate_components)])):
                        if self.loss_type == "kl-divergence":
                            B_numerator = torch.div((M * y_b), torch.clamp(M * (B_b @ H_b), min=self.eps)) @ H_b.T
                            B_denominator = torch.ones_like(y_b) @ H_b.T
                        else:
                            B_numerator = (M * y_b) @ H_b.T
                            B_denominator = (M * (B_b @ H_b)) @ H_b.T

                        Bs[b] *= torch.div(torch.clamp(B_numerator, min=self.eps), torch.clamp(B_denominator, min=self.eps))


                    # update partial H
                    H_label_numerator = torch.zeros_like(H_sub) + self.eps
                    H_label_denominator = torch.zeros_like(H_sub) + self.eps

                    start_idx = 0
                    for b, (M, y_b, B_b, H_b) in enumerate(zip(M_y_sub, y_sub, Bs, Hs_sub[:len(self.n_covariate_components)])):
                        end_idx = start_idx + H_b.shape[0]
                        if self.loss_type == "kl-divergence":
                            H_numerator = self.lam[b] * B_b.T @ torch.div((M * y_b), torch.clamp(M *(B_b @ H_b), min=self.eps))
                            H_denominator = self.lam[b] * B_b.T @ torch.ones_like(y_b)
                        else:
                            H_numerator = 2 * self.lam[b] * B_b.T @ (M * y_b)
                            H_denominator = 2 * self.lam[b] * B_b.T @ (M * (B_b @ H_b))

                        H_label_numerator[start_idx:end_idx] += H_numerator
                        H_label_denominator[start_idx:end_idx] += H_denominator
                        start_idx = end_idx

                    # update H
                    W = torch.cat(Ws, dim=1)
                    H_bio_numerator = 2 * W.T @ X_sub
                    H_bio_denominator = 2 * W.T @ W @ H_sub

                    numerator = torch.clamp(H_bio_numerator + H_label_numerator, min=self.eps)
                    denominator = torch.clamp(H_bio_denominator + H_label_denominator, min=self.eps)
                    H_sub *= torch.div(numerator, denominator)

                    start_idx = 0
                    for idx, h in enumerate(Hs):
                        end_idx = start_idx + h.shape[0]
                        h[:, indices] = H_sub[start_idx:end_idx]
                        start_idx = end_idx


                # loss calculation
                # all the loss calculate on cpu
                W = torch.cat(Ws, dim=1)
                H = torch.cat(Hs, dim=0)
                recon_loss = torch.norm(X - W @ H, 'fro')**2

                if self.loss_type == "kl-divergence":
                    label_loss = [self._kl_divergence(_y, _b @ _h) for _y, _b, _h in zip(y, Bs, Hs)]
                else:
                    label_loss = [torch.norm(_y - _b @ _h, 'fro')**2 for _y, _b, _h in zip(y, Bs, Hs)]

                weighted_label_loss = [self.lam[i] * ll for i, ll in enumerate(label_loss)]
                label_total_loss = torch.sum(torch.stack(weighted_label_loss))
                loss = recon_loss + label_total_loss
                loss_history.append([recon_loss.item()] + [l.item() for l in weighted_label_loss] + [loss.item()])

                # update lambda
                # self.lam = [torch.clamp(self.lam[i]-0.01*torch.log(1 + label_loss[i]), min=self.eps).item() for i in range(len(self.lam))]

                if verbose:
                    pbar.set_postfix({"objective loss": loss.item()})
                    pbar.update(1)

        Ws = [_w.cpu().numpy() for _w in Ws]
        Hs = [_h.cpu().numpy() for _h in Hs]
        Bs = [_b.cpu().numpy() for _b in Bs]

        # Save the loss history
        self.loss_history = pd.DataFrame(
            loss_history,
            columns=["reconstruction_loss"] +
                    [f"prediction_loss_{i+1}" for i in range(len(self.n_covariate_components))] +
                    ["total_loss"]
        )
        return Ws, Hs, Bs


    def _transform_wo_labels(
        self, X: ad.AnnData,
        n_iter: Optional[int] = None,
        batch_size = None,
        lam = None        
    ):
        """
        Perform data transformation without labels using non-negative matrix factorization (NMF).
        This method applies NMF to transform the input data `X` into a lower-dimensional representation
        using pre-trained weight matrices (`Ws`). The transformed data is stored in the `obsm` and `varm`
        attributes of the input AnnData object.
        Args:
            X (ad.AnnData): The input AnnData object containing the data to be transformed.
            n_iter (Optional[int], optional): The number of iterations for the NMF update rule. Defaults to None.
            batch_size (optional): Not used in this method. Defaults to None.
            lam (optional): Not used in this method. Defaults to None.
        Returns:
            Tuple[List[np.ndarray], Any]: 
                - A list of transformed matrices (`Hs_new`) corresponding to each condition.
                - An additional return value (`_`), which is not explicitly documented in this method.
        Notes:
            - The method initializes random matrices for the latent factors (`H`) and updates them iteratively
              using the multiplicative update rule for NMF.
            - The transformed matrices are stored in the `obsm` and `varm` attributes of the input AnnData object
              under keys corresponding to the condition names and the "ALPINE_embedding".
            - The method ensures reproducibility by setting the random seed if `self.random_state` is provided.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        y = self.condition_names
        X_new, _, _ = self._process_input(X, y)
        X_new = torch.tensor(X_new, dtype=torch.float32, device=self.device)

        Ws = [torch.tensor(w, dtype=torch.float32, device=self.device) for w in self.Ws]
        W = torch.cat(Ws, dim=1)

        n_sample = X_new.shape[1]
        Hs_new = [torch.rand((k, n_sample), dtype=torch.float32, device=self.device) for k in self.n_all_components]
        H = torch.cat(Hs_new, dim=0)

        for i in range(n_iter):
            numerator = torch.clamp(2 * W.T @ X_new, min=self.eps)
            denominator = torch.clamp(2 * W.T @ W @ H, min=self.eps)
            H *= torch.div(numerator, denominator)
            H = torch.clamp(H, min=self.eps)
        
        start_idx = 0
        for idx, h in enumerate(Hs_new):
            end_idx = start_idx + h.shape[0]
            Hs_new[idx] = H[start_idx:end_idx]
            start_idx = end_idx

        Hs_new = [h.cpu().numpy() for h in Hs_new]
        for i, y_name in enumerate(y):
            X.obsm[y_name] = Hs_new[i].T
            X.varm[y_name] = deepcopy(self.Ws[i])
        X.obsm["ALPINE_embedding"] = deepcopy(Hs_new[-1].T)
        X.varm["ALPINE_embedding"] = deepcopy(self.Ws[-1])

        return Hs_new, _


    def _transform(
            self, X: ad.AnnData, 
            n_iter: Optional[int] = None,
            batch_size = None,
            lam = None
        ):

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        y = self.condition_names
        X_new, y_new_input, _ = self._process_input(X, y)
        y_new, M_y = zip(*[self._to_dummies_from_train(i, yi) for i, yi in enumerate(y_new_input)])
        M_y = [torch.tensor(m, dtype=torch.float32, device=self.device) for m in M_y]

        X_new = torch.tensor(X_new, dtype=torch.float32, device=self.device)
        y_new = [torch.tensor(yi, dtype=torch.float32, device=self.device) for yi in y_new]
        Bs = [torch.tensor(b, dtype=torch.float32, device=self.device) for b in self.Bs]
        Ws = [torch.tensor(w, dtype=torch.float32, device=self.device) for w in self.Ws]
        W = torch.cat(Ws, dim=1)

        n_sample = X_new.shape[1]
        Hs_new = [torch.rand((k, n_sample), dtype=torch.float32, device=self.device) for k in self.n_all_components]
        H_new = torch.cat(Hs_new, dim=0)

        if lam is None:
            W = torch.cat(Ws, dim=1)
            H_new = torch.cat(Hs_new, dim=0)
            recon_loss = torch.norm(X_new - W @ H_new, 'fro')**2
            if self.loss_type == "kl-divergence":
                label_loss = [self._kl_divergence(_y, _b @ _h) for _y, _b, _h in zip(y_new, Bs, Hs_new[:len(self.n_covariate_components)])]
                lam = [(recon_loss.item() / 1e+3) / ll.clamp(min=self.eps).item() for ll in label_loss]
            else:
                label_loss = [torch.norm(_y - _b @ _h, 'fro')**2 for _y, _b, _h in zip(y_new, Bs, Hs_new[:len(self.n_covariate_components)])]
                lam = [(recon_loss.item() / 1e+2) / ll.clamp(min=self.eps).item() for ll in label_loss]

        loss_history = []
        if batch_size is None:
            batch_size = int(n_sample / 3)
        for _ in range(n_iter):
            for _ in range(0, n_sample, batch_size):
                indices = torch.randperm(n_sample, device=self.device)[:batch_size]

                X_sub = X_new[:, indices]
                Hs_sub = [h[:, indices] for h in Hs_new]
                y_sub = [_y[:, indices] for _y in y_new]
                H_sub = torch.cat(Hs_sub, dim=0)
                M_y_sub = [m[:, indices] for m in M_y]

                H_label_numerator = torch.zeros_like(H_sub, device=self.device)
                H_label_denominator = torch.zeros_like(H_sub, device=self.device)
                
                start_idx = 0
                for b, (M, y_b, B_b, H_b) in enumerate(zip(M_y_sub, y_sub, Bs, Hs_sub[:len(self.n_covariate_components)])):
                    end_idx = start_idx + H_b.shape[0]
                    if self.loss_type == "kl-divergence":
                        H_numerator = lam[b] * B_b.T @ torch.div(M * y_b, torch.clamp(M * (B_b @ H_b), min=self.eps))
                        H_denominator = lam[b] * B_b.T @ torch.ones_like(y_b)
                    else:
                        H_numerator = 2 * lam[b] * B_b.T @ (M * y_b)
                        H_denominator = 2 * lam[b] * B_b.T @ (M * (B_b @ H_b))
                    
                    H_label_numerator[start_idx:end_idx] += H_numerator
                    H_label_denominator[start_idx:end_idx] += H_denominator
                    start_idx = end_idx

                # Update H
                numerator = torch.clamp(2 * W.T @ X_sub + H_label_numerator, min=self.eps)
                denominator = torch.clamp(2 * W.T @ W @ H_sub + H_label_denominator, min=self.eps)

                H_sub *= torch.div(numerator, denominator)
                
                start_idx = 0
                for idx, h in enumerate(Hs_new):
                    end_idx = start_idx + h.shape[0]
                    Hs_new[idx][:, indices] = H_sub[start_idx:end_idx]
                    start_idx = end_idx

            # loss calculation
            H_new = torch.cat(Hs_new, dim=0)
            recon_loss = torch.norm(X_new - W @ H_new, 'fro')**2

            if self.loss_type == "kl-divergence":
                label_loss = [self._kl_divergence(_y, _b @ _h) for _y, _b, _h in zip(y_new, Bs, Hs_new)]
            else:
                label_loss = [torch.norm(_y - _b @ _h, 'fro')**2 for _y, _b, _h in zip(y_new, Bs, Hs_new)]

            label_loss = [lam[i] * ll for i, ll in enumerate(label_loss)]
            label_total_loss = torch.sum(torch.stack(label_loss))
            loss = recon_loss + label_total_loss
            loss_history.append([recon_loss.item()] + [l.item() for l in label_loss] + [loss.item()])
        
        loss_history = pd.DataFrame(
            loss_history,
            columns=["reconstruction_loss"] +
                    [f"prediction_loss_{i+1}" for i in range(len(self.n_covariate_components))] +
                    ["total_loss"]
        )
        
        Hs_new = [h.cpu().numpy() for h in Hs_new]
        for i, y_name in enumerate(y):
            X.obsm[y_name] = Hs_new[i].T
            X.varm[y_name] = deepcopy(self.Ws[i])
        X.obsm["ALPINE_embedding"] = deepcopy(Hs_new[-1].T)
        X.varm["ALPINE_embedding"] = deepcopy(self.Ws[-1])
        
        return Hs_new, loss_history