import scanpy as sc
import numpy as np
import pandas as pd
import pickle

from typing import Tuple
from kneed import KneeLocator
from sklearn.metrics.cluster import adjusted_rand_score, homogeneity_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from .main import ALPINE

class ComponentOptimizer:

    def __init__(
            self,
            adata,
            covariate_keys,
            loss_type='kl-divergence',
            max_iter=None,
            batch_size=None,
            gpu=True,
            random_state=None,
            lam=None,
        ):
        self.adata = adata.copy()
        self.covariate_keys = covariate_keys
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.random_state = random_state
        self.gpu = gpu
        self.batch_size = batch_size
        self.lam = lam
        self.best_param = {}
        self.minimum_set_param = {}

        if self.max_iter is None:
            self.max_iter_detect = True

    

    def calc_ari(self, args):
        n_covariate_components = args['n_covariate_components']
        n_components = args['n_components']
        lam = [10**lam for lam in args['lam']]
        alpha_W = args['alpha_W']
        orth_W = args['orth_W']
        l1_ratio = args['l1_ratio']

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

        # Fit the model on the training data
        adata = self.adata.copy()
        model.fit(X=adata, covariate_keys=self.covariate_keys, max_iter=self.max_iter, batch_size=self.batch_size, verbose=False)
        model.store_embeddings(adata)
        self.max_iter = model.max_iter

        sc.pp.neighbors(adata, use_rep='ALPINE_embedding')
        sc.tl.leiden(adata, flavor="igraph")

        # Calculate the score
        score = 0
        for key in self.covariate_keys:
            selected_idx = ~adata.obs[key].isna()
            score += adjusted_rand_score(adata.obs[key][selected_idx], adata.obs["leiden"][selected_idx])
            score += homogeneity_score(adata.obs[key][selected_idx], adata.obs["leiden"][selected_idx])
        score /= len(self.covariate_keys)
        
        return score


    def n_component_assignment (self, x, min_components):
        n_all_components = int(x["n_all_components"])

        n_component_ratio = [x[f'components_ratio_{i}'] for i in range(len(self.covariate_keys) + 1)]
        n_component_ratio = [ratio / sum(n_component_ratio) for ratio in n_component_ratio]

        n_components = int(x['n_all_components'] / 2)
        rest_components = n_all_components - n_components
        n_component_int = [int(ratio * rest_components) for ratio in n_component_ratio]

        n_components += n_component_int[0]
        n_covariate_components = n_component_int[1:]
        n_covariate_components = [max(min_components[i], n) for i, n in enumerate(n_covariate_components)]

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
            'max_iter': self.max_iter,
            'score': score
        }
        
        if self.max_iter_detect:
            if len(self.iter_records) < 5:
                self.iter_records.append(self.max_iter)
                self.max_iter = None
            else:
                self.max_iter = max(self.iter_records)

        loss = score + self.weight_reduce_covar_dims * (sum(n_covariate_components)/(sum(n_covariate_components) + n_components))
        return {'loss': loss, 'status': STATUS_OK, 'params': trial_history}
    

    def bayesian_search(
            self,
            n_total_components_range=(50, 100),
            lam_power_range=(2, 6),
            alpha_W_range=(0, 1),
            orth_W_range=(0, 0.5),
            l1_ratio_range=(0, 1),
            weight_reduce_covar_dims = 0,
            max_evals=50,
            min_components=None,
            trials_filename=None
        ):

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

        # Sort by score in descending order and reset the index
        history_df = history_df.sort_values(by='score', ascending=False).reset_index(drop=True)

        return history_df



    def fit_the_best_param(self):

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
