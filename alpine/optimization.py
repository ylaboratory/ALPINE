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
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
import warnings

class ComponentOptimizer:
    
    def __init__(
            self,
            adata,
            covariate_keys,
            loss_type='kl-divergence',
            max_iter=None,
            batch_size=None,
            device="cuda",
            random_state=None
        ):
        self.adata = adata.copy()
        self.covariate_keys = covariate_keys
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.random_state = random_state
        self.device = device
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
                    device=self.device,
                )
                
                model.fit(X=train_adata, covariate_keys=self.covariate_keys, max_iter=self.max_iter, batch_size=self.batch_size, verbose=False)
                model.store_embeddings(train_adata)
                _ = model.transform(val_adata, use_label=False)

                sc.pp.neighbors(val_adata, use_rep='ALPINE_embedding')
                # scores.append(self.calc_ARI_score(val_adata))

                sc.tl.leiden(val_adata, flavor="igraph", resolution=5)
                score = 0
                for key in self.covariate_keys:
                    selected_idx = ~val_adata.obs[key].isna()
                    score += adjusted_rand_score(val_adata.obs[key][selected_idx], val_adata.obs["leiden"][selected_idx])
                    score += homogeneity_score(val_adata.obs[key][selected_idx], val_adata.obs["leiden"][selected_idx])
                score /= len(self.covariate_keys)

                scores.append(score)


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
                device=self.device,
            )

            adata = self.adata.copy()
            model.fit(X=adata, covariate_keys=self.covariate_keys, max_iter=self.max_iter, batch_size=self.batch_size, verbose=False)
            model.store_embeddings(adata)

            sc.pp.neighbors(adata, use_rep='ALPINE_embedding')
            # score = self.calc_ARI_score(adata)
            sc.tl.leiden(adata, flavor="igraph", resolution=1)
            score = 0
            for key in self.covariate_keys:
                selected_idx = ~adata.obs[key].isna()
                score += adjusted_rand_score(adata.obs[key][selected_idx], adata.obs["leiden"][selected_idx])
                score += homogeneity_score(adata.obs[key][selected_idx], adata.obs["leiden"][selected_idx])
            score /= len(self.covariate_keys)
        
            if self.max_iter_detect:
                self.iter_records.append(model.max_iter)

        return score

    def calc_ARI_score(self, adata, n=10):

        def score_for_res(res):
            adata_copy = adata.copy()
            sc.tl.leiden(adata_copy, flavor="igraph", resolution=res)
            score = 0
            for key in self.covariate_keys:
                selected_idx = ~adata_copy.obs[key].isna()
                score += adjusted_rand_score(adata_copy.obs[key][selected_idx], adata_copy.obs["leiden"][selected_idx])
                score += homogeneity_score(adata_copy.obs[key][selected_idx], adata_copy.obs["leiden"][selected_idx])
            score /= len(self.covariate_keys)
            return score

        resolutions = [2 * x / n for x in range(1, n + 1)]
        scores = Parallel(n_jobs=-1)(delayed(score_for_res)(res) for res in resolutions)
        return min(scores)


    def objective(self, space):
        # Handle multiple covariate components separately
        lam = [space[f'lam_{i}'] for i in range(len(self.covariate_keys))]
        n_components = int(space['n_components'])
        n_covariate_components = [int(space[f'n_covariate_components_{i}']) for i in range(len(self.covariate_keys))]

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
    

    def search_hyperparams(
            self,
            n_components_range=(10, 50),
            max_covariate_components=None,
            lam_power_range=(1, 6),
            alpha_W_range=(0, 100),
            orth_W_range=(0, 0.5),
            l1_ratio_range=(0, 1),
            weight_reduce_covar_dims=0.0,
            n_splits=None,
            max_evals=100,
            trials_filename=None
        ):

        if max_covariate_components is None:
            max_covariate_components = [self.adata.obs[key].nunique() * 2 for key in self.covariate_keys]
            warnings.warn(
                f"`max_covariate_components` was not provided. Defaulting to 2 × the number of unique values for each covariate: {max_covariate_components}. "
                "If this does not yield optimal results, please set `max_covariate_components` manually for each covariate. "
                "For further tuning advice, consult the documentation or visit the GitHub repository.",
                UserWarning
            )
        else:
            if isinstance(max_covariate_components, list):
                if len(max_covariate_components) != len(self.covariate_keys):
                    raise ValueError("max_covariate_components should have the same length as the number of covariates.")
        
        if any(comp < 2 for comp in max_covariate_components):
            raise ValueError("max_covariate_components should be greater than or equal to 2.")
        
        self.max_covariate_components = max_covariate_components
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
            'n_components': hp.quniform('n_components', n_components_range[0], n_components_range[1], 1),
            'alpha_W': hp.uniform('alpha_W', alpha_W_range[0], alpha_W_range[1]),
            'orth_W': hp.uniform('orth_W', orth_W_range[0], orth_W_range[1]),
            'l1_ratio': hp.uniform('l1_ratio', l1_ratio_range[0], l1_ratio_range[1])
        }

        # Distribute the remaining space across covariate components
        for i in range(len(self.covariate_keys)):
            self.space[f'lam_{i}'] = hp.uniform(f'lam_{i}', lam_power_range[0], lam_power_range[1])
            self.space[f'n_covariate_components_{i}'] = hp.quniform(f'n_covariate_components_{i}', 1.9, max_covariate_components[i], 1)

        # Run the optimization using TPE
        best = fmin(self.objective, self.space, algo=tpe.suggest, max_evals=max_evals + len(self.trials.trials), trials=self.trials)

        n_components = int(best['n_components'])
        n_covariate_components = [int(best[f'n_covariate_components_{i}']) for i in range(len(self.covariate_keys))]

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

        n_components = int(self.space['n_components'])
        n_covariate_components = [int(self.space[f'n_covariate_components_{i}']) for i in range(len(self.covariate_keys))]

        # Assign the best parameters from the latest search
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
            device = self.device,
            
        )
        model.fit(X=self.adata, covariate_keys=self.covariate_keys, max_iter=self.max_iter, batch_size=self.batch_size, verbose=False)
        return model
