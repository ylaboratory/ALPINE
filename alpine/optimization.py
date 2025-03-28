import scanpy as sc
import numpy as np
import pandas as pd
import pickle

from typing import Tuple
from kneed import KneeLocator
from sklearn.metrics.cluster import adjusted_rand_score, homogeneity_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from .main import ALPINE
from sklearn.model_selection import StratifiedKFold

class ComponentOptimizer:
    """
    ComponentOptimizer is a class designed for optimizing the parameters of the ALPINE model using Bayesian optimization. 
    It provides methods for calculating ARI scores, assigning components, performing Bayesian searches, and managing trials.
    Attributes:
        adata (AnnData): The annotated data matrix.
        covariate_keys (list): List of covariate keys used for stratification.
        loss_type (str): The type of loss function to use. Default is 'kl-divergence'.
        max_iter (int or None): Maximum number of iterations for model training. If None, it will be detected dynamically.
        batch_size (int or None): Batch size for training. Default is None.
        gpu (bool): Whether to use GPU for training. Default is True.
        random_state (int or None): Random seed for reproducibility. Default is None.
        best_param (dict): Stores the best parameters found during optimization.
        minimum_set_param (dict): Stores the minimum set of parameters.
        min_components (list): Minimum number of components for each covariate.
        weight_reduce_covar_dims (float): Weight for penalizing the number of covariate components.
        iter_records (list): Records of iterations during training.
        n_splits (int or None): Number of splits for cross-validation.
        trials (Trials): Stores the trials object for Bayesian optimization.
        space (dict): Search space for Bayesian optimization.
    Methods:
        calc_ari(args):
            Calculates the Adjusted Rand Index (ARI) and homogeneity score for the given parameters.
        n_component_assignment(x, min_components):
            Assigns the number of components for the model based on the given ratios and minimum components.
        objective(space):
            Objective function for Bayesian optimization. Calculates the loss based on ARI and penalizes covariate dimensions.
        bayesian_search(n_total_components_range, lam_power_range, alpha_W_range, orth_W_range, l1_ratio_range, 
                        weight_reduce_covar_dims, n_splits, max_evals, min_components, trials_filename):
            Performs Bayesian optimization to find the best hyperparameters for the model.
        extend_training(extra_evals):
            Extends the Bayesian optimization process with additional evaluations.
        save_trials(filename):
            Saves the current trials to a file.
        load_trials(filename):
            Loads trials from a file.
        get_hyperparameter(idx):
            Retrieves the hyperparameters of a specific trial by index.
        get_train_history():
            Retrieves the training history as a pandas DataFrame, including trial details and scores.
        fit_the_best_param():
            Fits the ALPINE model using the best parameters found during Bayesian optimization.
    """
    def __init__(
            self,
            adata,
            covariate_keys,
            loss_type='kl-divergence',
            max_iter=None,
            batch_size=None,
            gpu=True,
            random_state=None
        ):
        self.adata = adata.copy()
        self.covariate_keys = covariate_keys
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.random_state = random_state
        self.gpu = gpu
        self.batch_size = batch_size
        self.best_param = {}
        self.minimum_set_param = {}

        if self.max_iter is None:
            self.max_iter_detect = True
        else:
            self.max_iter_detect = False

    

    def calc_ari(self, args):
        n_covariate_components = args['n_covariate_components']
        n_components = args['n_components']
        lam = [10**lam for lam in args['lam']]
        alpha_W = args['alpha_W']
        orth_W = args['orth_W']
        l1_ratio = args['l1_ratio']

        # Create joint labels for stratification
        if len(self.covariate_keys) == 1:
            joint_labels = self.adata.obs[self.covariate_keys[0]].astype(str)
        else:
            joint_labels = self.adata.obs[self.covariate_keys[0]].astype(str)
            for key in self.covariate_keys[1:]:
                joint_labels = joint_labels + '_' + self.adata.obs[key].astype(str)

        scores = []
        
        if self.n_splits is not None:
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            
            for train_idx, val_idx in skf.split(self.adata.X, joint_labels):
                train_adata = self.adata[train_idx].copy()
                val_adata = self.adata[val_idx].copy()
                
                model = ALPINE(
                    n_covariate_components=n_covariate_components,
                    n_components=n_components,
                    lam=lam,
                    alpha_W=alpha_W,
                    orth_W=orth_W,
                    l1_ratio=l1_ratio,
                    random_state=self.random_state,
                    loss_type=self.loss_type,
                    gpu=self.gpu,
                )
                
                model.fit(X=train_adata, covariate_keys=self.covariate_keys, max_iter=self.max_iter, batch_size=self.batch_size, verbose=False)
                model.store_embeddings(train_adata)
                _ = model.transform(val_adata, use_label=False)

                sc.pp.neighbors(val_adata, use_rep='ALPINE_embedding')
                sc.tl.leiden(val_adata, flavor="igraph")

                fold_score = 0
                for key in self.covariate_keys:
                    selected_idx = ~val_adata.obs[key].isna()
                    fold_score += adjusted_rand_score(val_adata.obs[key][selected_idx], val_adata.obs["leiden"][selected_idx])
                    fold_score += homogeneity_score(val_adata.obs[key][selected_idx], val_adata.obs["leiden"][selected_idx])
                scores.append(fold_score / len(self.covariate_keys))
                
                if self.max_iter_detect:
                    self.iter_records.append(model.max_iter)

            score = np.mean(scores)
        
        else:
            model = ALPINE(
                n_covariate_components=n_covariate_components,
                n_components=n_components,
                lam=lam,
                alpha_W=alpha_W,
                orth_W=orth_W,
                l1_ratio=l1_ratio,
                random_state=self.random_state,
                loss_type=self.loss_type,
                gpu=self.gpu,
            )

            adata = self.adata.copy()
            model.fit(X=adata, covariate_keys=self.covariate_keys, max_iter=self.max_iter, batch_size=self.batch_size, verbose=False)
            model.store_embeddings(adata)

            sc.pp.neighbors(adata, use_rep='ALPINE_embedding')
            sc.tl.leiden(adata, flavor="igraph")

            score = 0
            for key in self.covariate_keys:
                selected_idx = ~adata.obs[key].isna()
                score += adjusted_rand_score(adata.obs[key][selected_idx], adata.obs["leiden"][selected_idx])
                score += homogeneity_score(adata.obs[key][selected_idx], adata.obs["leiden"][selected_idx])
            score /= len(self.covariate_keys)

            if self.max_iter_detect:
                self.iter_records.append(model.max_iter)

        return score



    def n_component_assignment(self, x, min_components):
        n_all_components = int(x["n_all_components"])

        n_component_ratio = [x[f'components_ratio_{i}'] for i in range(len(self.covariate_keys) + 1)]
        n_component_ratio = [ratio / sum(n_component_ratio) for ratio in n_component_ratio]

        n_components = int(n_all_components / 2)
        rest_components = n_all_components - n_components
        n_covariate_components = [int(ratio * rest_components) for ratio in n_component_ratio[1:]]

        # Ensure that each covariate component is at least the minimum required
        n_covariate_components = [max(min_components[i], n) for i, n in enumerate(n_covariate_components)]

        # Adjust n_components to ensure the total sum is equal to n_all_components
        total_covariate_components = sum(n_covariate_components)
        n_components = n_all_components - total_covariate_components

        return n_components, n_covariate_components


    def objective(self, space):
        # Handle multiple covariate components separately
        lam = [space[f'lam_{i}'] for i in range(len(self.covariate_keys))]
        n_components, n_covariate_components = self.n_component_assignment(space, self.min_components)

        args = {
            'n_components': n_components,
            'n_covariate_components': n_covariate_components,
            'lam': lam,
            'alpha_W': space['alpha_W'],
            'orth_W': space['orth_W'],
            'l1_ratio': space['l1_ratio']
        }

        score = self.calc_ari(args)
        
        trial_history = {
            'n_components': n_components,
            'n_covariate_components': n_covariate_components,
            'lam': [10**l for l in lam],
            'alpha_W': space['alpha_W'],
            'orth_W': space['orth_W'],
            'l1_ratio': space['l1_ratio'],
            'max_iter': self.iter_records[-1] if self.max_iter_detect else self.max_iter,
            'score': score
        }
        
        if self.max_iter_detect:
            if len(self.iter_records) >= 5:
                self.max_iter = max(self.iter_records)

        n_totoal_covariate_components = sum(n_covariate_components)
        loss = score + self.weight_reduce_covar_dims * (n_totoal_covariate_components/(n_totoal_covariate_components + n_components))
        return {'loss': loss, 'status': STATUS_OK, 'params': trial_history}
    

    def bayesian_search(
            self,
            n_total_components_range=(50, 100),
            lam_power_range=(1, 6),
            alpha_W_range=(0, 1),
            orth_W_range=(0, 0.5),
            l1_ratio_range=(0, 1),
            weight_reduce_covar_dims=0.5,
            n_splits=None,
            max_evals=50,
            min_components=None,
            trials_filename=None
        ):
        """
        Perform Bayesian optimization to search for the best hyperparameters.
        This method uses Tree-structured Parzen Estimator (TPE) to optimize the 
        hyperparameters for a model. It defines a search space, evaluates the 
        objective function, and finds the best parameters.
        Parameters:
            n_total_components_range (tuple, optional): 
                Range for the total number of components. Defaults to (50, 100).
            lam_power_range (tuple, optional): 
                Range for the power of lambda regularization. Defaults to (1, 6).
            alpha_W_range (tuple, optional): 
                Range for the alpha_W parameter. Defaults to (0, 1).
            orth_W_range (tuple, optional): 
                Range for the orthogonality regularization of W. Defaults to (0, 0.5).
            l1_ratio_range (tuple, optional): 
                Range for the L1 ratio. Defaults to (0, 1).
            weight_reduce_covar_dims (float, optional): 
                Weight to reduce covariance dimensions. Defaults to 0.5.
            n_splits (int, optional): 
                Number of splits for cross-validation. Defaults to None.
            max_evals (int, optional): 
                Maximum number of evaluations for the optimization. Defaults to 50.
            min_components (int or list, optional): 
                Minimum number of components for each covariate. If None, it is 
                inferred from the data. Defaults to None.
            trials_filename (str, optional): 
                Path to a file to load/save optimization trials. Defaults to None.
        Returns:
            dict: A dictionary containing the best hyperparameters:
                - 'n_components': Total number of components.
                - 'n_covariate_components': Number of components for each covariate.
                - 'lam': List of lambda values for each covariate.
                - 'alpha_W': Best alpha_W value.
                - 'orth_W': Best orth_W value.
                - 'l1_ratio': Best L1 ratio value.
                - 'random_state': Random state used for reproducibility.
        Raises:
            ValueError: If `min_components` is less than 2 or if its length does 
                not match the number of covariates.
        """

        if min_components is None:
            min_components = [self.adata.obs[key].nunique() for key in self.covariate_keys]
        else:
            if isinstance(min_components, list):
                if len(min_components) != len(self.covariate_keys):
                    raise ValueError("min_components should have the same length as the number of covariates.")
            else:
                min_components = [min_components] * len(self.covariate_keys)
        
        if any(comp < 2 for comp in min_components):
            raise ValueError("min_components should be greater than or equal to 2.")
        self.min_components = min_components
        self.weight_reduce_covar_dims = weight_reduce_covar_dims
        self.iter_records = []
        self.n_splits = n_splits

        # Load previous trials if specified
        if trials_filename is not None:
            self.load_trials(trials_filename)
        else:
            self.trials = Trials()  # Create a new trials object if not loading

        # Define the search space for Bayesian Optimization
        self.space = {
            # Ensure n_components is at least 50% of n_total_components
            'n_all_components': hp.quniform('n_all_components', n_total_components_range[0], n_total_components_range[1], 1),
            'alpha_W': hp.uniform('alpha_W', alpha_W_range[0], alpha_W_range[1]),
            'orth_W': hp.uniform('orth_W', orth_W_range[0], orth_W_range[1]),
            'l1_ratio': hp.uniform('l1_ratio', l1_ratio_range[0], l1_ratio_range[1])
        }

        # Distribute the remaining space across covariate components
        for i in range(len(self.covariate_keys)):
            self.space[f'lam_{i}'] = hp.uniform(f'lam_{i}', lam_power_range[0], lam_power_range[1])
            # Ensure that sum of all covariates equals the remaining space
        
        # covariate
        for i in range(len(self.covariate_keys) + 1):
            self.space[f'components_ratio_{i}'] = hp.uniform(f'components_ratio_{i}', 0, 1)

        # Run the optimization using TPE
        best = fmin(self.objective, self.space, algo=tpe.suggest, max_evals=max_evals + len(self.trials.trials), trials=self.trials)

        n_components, n_covariate_components = self.n_component_assignment(best, self.min_components)
        self.best_param['n_components'] = n_components
        self.best_param['n_covariate_components'] = n_covariate_components
        self.best_param['lam'] = [float(10**best[f'lam_{i}']) for i in range(len(self.covariate_keys))]
        self.best_param['alpha_W'] = best['alpha_W']
        self.best_param['orth_W'] = best['orth_W']
        self.best_param['l1_ratio'] = best['l1_ratio']
        self.best_param['random_state'] = self.random_state

        return self.best_param


    def extend_training(self, extra_evals=50):
        """
        Continue the Bayesian optimization process with more evaluations.

        Parameters:
            extra_evals: int
                The number of additional evaluations to perform during the Bayesian search.

        Returns:
            dict
                The updated best parameters after the additional evaluations.
        """
        if not hasattr(self, 'trials'):
            raise RuntimeError("Please run bayesian_search() before extending training.")
        
        # Update the max_evals for continuing the optimization
        best = fmin(
            fn=self.objective,
            space=self.space,
            algo=tpe.suggest,
            max_evals=len(self.trials.trials) + extra_evals,  # Continue from the current trial count
            trials=self.trials
        )


        # Assign the best parameters from the latest search
        n_components, n_covariate_components = self.n_component_assignment(best, self.min_components)
        self.best_param['n_components'] = n_components
        self.best_param['n_covariate_components'] = n_covariate_components
        self.best_param['lam'] = [float(10**best[f'lam_{i}']) for i in range(len(self.covariate_keys))]
        self.best_param['alpha_W'] = best['alpha_W']
        self.best_param['orth_W'] = best['orth_W']
        self.best_param['l1_ratio'] = best['l1_ratio']
        self.best_param['random_state'] = self.random_state
        
        return self.best_param



    def save_trials (self, filename: str):
        """
        Save the current trials to a file.
        
        Parameters:
            filename: str
                Path to the file where the trials will be saved.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.trials, f)
        print(f"Trials saved to {filename}")


    def load_trials (self, filename: str):
        """
        Load the trials from a file.
        
        Parameters:
            filename: str
                Path to the file from which to load the trials.
        """
        with open(filename, 'rb') as f:
            self.trials = pickle.load(f)
        print(f"Trials loaded from {filename}")


    def get_hyperparameter(self, idx):
        """
        Retrieve the hyperparameters of a specific trial based on its index in the sorted training history.
        Args:
            idx (int): The index of the trial in the sorted training history.
        Returns:
            dict: A dictionary containing the hyperparameters of the specified trial.
        Raises:
            IndexError: If the provided index is out of bounds for the training history.
            KeyError: If the expected keys ('tid' or 'params') are missing in the trial data.
        Notes:
            - The training history is assumed to be sorted prior to fetching the trial.
            - The method matches the trial in `self.trials` using the trial ID ('tid') from the history.
        """
        # Fetch the sorted trial history
        history_df = self.get_train_history()

        # Retrieve the corresponding trial from sorted index
        trial = history_df.iloc[idx]

        # Find the corresponding trial in self.trials using the 'tid' (trial ID)
        trial_tid = trial['tid']
        
        # Extract the trial's hyperparameters using the 'tid'
        for trial in self.trials.trials:
            if trial['tid'] == trial_tid:
                return trial['result']['params']

    def get_train_history(self):
        """
        Retrieves and processes the training history from the optimization trials.

        This method extracts trial information, processes it into a structured format,
        and returns a pandas DataFrame containing details about each trial, including
        the parameters, scores, and derived metrics.

        Returns:
            pd.DataFrame: A DataFrame containing the following columns:
                - n_components: The number of components used in the trial.
                - n_covariate_components_X: Expanded columns for each covariate component (X is the index).
                - n_total_components: The total number of components, calculated as the sum of 
                  `n_components` and all `n_covariate_components`.
                - score: The score (loss) of the trial.
                - tid: The trial ID for tracking.
                - Other columns corresponding to the trial parameters.

        Notes:
            - The DataFrame is sorted by the `score` column in descending order.
            - The index of the DataFrame is reset after sorting.
            - The columns are reordered to place `n_components` and related columns at the beginning.
        """
        history = []

        # Extract trial information
        for trial in self.trials.trials:
            if 'result' in trial and trial['result']['status'] == STATUS_OK:
                trial_info = trial['result']['params']
                trial_info['score'] = trial['result']['loss']
                trial_info['tid'] = trial['tid']  # Add trial ID for tracking
                history.append(trial_info)

        # Convert to pandas DataFrame
        history_df = pd.DataFrame(history)

        # Expand the list columns into separate columns
        n_covariate_df = pd.DataFrame(history_df['n_covariate_components'].tolist(), 
                                    columns=[f'n_covariate_components_{i}' for i in range(len(history_df['n_covariate_components'].iloc[0]))])
        lam_df = pd.DataFrame(history_df['lam'].tolist(), 
                            columns=[f'lam_{i}' for i in range(len(history_df['lam'].iloc[0]))])

        # Concatenate the expanded columns back to the original DataFrame
        history_df = pd.concat([history_df.drop(columns=['n_covariate_components', 'lam']), n_covariate_df, lam_df], axis=1)

        # Add n_total_components as the sum of n_components and all n_covariate_components
        history_df['n_total_components'] = history_df['n_components'] + history_df[[f'n_covariate_components_{i}' for i in range(len(n_covariate_df.columns))]].sum(axis=1)

        # Reorder columns to have n_components related columns at the beginning
        columns_order = ['n_components'] + [f'n_covariate_components_{i}' for i in range(len(n_covariate_df.columns))] + ['n_total_components'] + [col for col in history_df.columns if col not in ['n_components', 'n_total_components'] + [f'n_covariate_components_{i}' for i in range(len(n_covariate_df.columns))]]
        history_df = history_df[columns_order]

        # Sort by score in descending order and reset the index
        history_df = history_df.sort_values(by='score', ascending=False).reset_index(drop=True)

        return history_df



    def fit_the_best_param(self):
        """
        Fits the ALPINE model using the best parameters found through Bayesian search.
        This method initializes an ALPINE model with the best parameters stored in `self.best_param`
        and fits it using the provided data and configuration.
        Returns:
            ALPINE: An instance of the ALPINE model fitted with the best parameters.
        Raises:
            RuntimeError: If the `best_param` attribute is not set. Ensure that `bayesian_search()` 
                          is run prior to calling this method to determine the best parameters.
        """

        if not hasattr(self, 'best_param'):
            raise RuntimeError("Please run bayesian_search() to find the best parameters first.")
        
        model = ALPINE (
            **self.best_param,
            random_state = self.random_state,
            loss_type = self.loss_type,
            gpu = self.gpu,
            
        )
        model.fit(X=self.adata, covariate_keys=self.covariate_keys, max_iter=self.max_iter, batch_size=self.batch_size, verbose=False)
        return model
