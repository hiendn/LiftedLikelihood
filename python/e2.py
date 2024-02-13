import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from scipy.integrate import quad
from functools import wraps as _wraps
from dataclasses import dataclass

from utils import *

def vectorized_dectorator(fn):
    @_wraps(fn)
    def wrapper(*args, **kwargs):
        return np.vectorize(fn)(*args, **kwargs)
    return wrapper

@vectorized_dectorator
def dtrue(x):
    if x < 0 or x > 1:
        return 0
    elif 0 <= x <= 0.5:
        return 2 - 4 * x
    elif 0.5 < x <= 1:
        return -2 + 4 * x
    
@dataclass
class E2:
    list_of_sample_sizes: list[int]
    list_of_mixture_components: list[int]
    
    @staticmethod
    def h_lifting_function(x):
        return beta.pdf(x, 1, 1)
    
    @staticmethod
    def select_initial_component_params(mixture_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if mixture_size == 1:
            al_vec = np.array([1])
            be_vec = np.array([1])
            pi_vec = np.array([1])
        else:
            all_means = np.linspace(0.1, 0.9, 16)
            all_means = all_means[np.argsort(-np.abs(all_means - 0.5))]
            means = all_means[:mixture_size]
            al_vec = means * (means * (1 - means) / 0.01 - 1)
            be_vec = (1 - means) * (means * (1 - means) / 0.01 - 1)
            pi_vec = np.repeat(1 / mixture_size, mixture_size)
            
        return al_vec, be_vec, pi_vec
    
    @staticmethod
    def sample_from_density(n) -> np.ndarray:
        LL = np.random.choice([1, 2], n, replace=True)
        XX = np.where(LL == 1, np.abs((np.random.uniform(size=n) + np.random.uniform(size=n)) - 1) / 2, 
                            -np.abs((np.random.uniform(size=n) + np.random.uniform(size=n)) - 1) / 2 + 1)
        return XX
    
    @staticmethod
    def sample_from_lifting_fn(n) -> np.ndarray:
        return np.random.uniform(0, 1, size=n)
    
    @staticmethod
    def fitted_loss(xx, alpha_, beta_, pi_) -> float:
        fitted = sum(pi_ * beta.pdf(xx, alpha_, beta_))
        fitted_with_lift = fitted + 1  #dunif(1,1)
        return fitted_with_lift
    
    @staticmethod
    def integrand_X(xx, al_vec, be_vec, pi_vec) -> np.ndarray:
        fitted_loss = E2.fitted_loss(xx, al_vec, be_vec, pi_vec)
        up = dtrue(xx)
        return np.log(fitted_loss)*up
    
    @staticmethod
    def integrand_Y(xx, al_vec, be_vec, pi_vec) -> np.ndarray:
        fitted_loss = E2.fitted_loss(xx, al_vec, be_vec, pi_vec)
        return np.log(fitted_loss)
    
    def optimize_params_for_fixed_mixture_components(self, XX, YY, sample_size, mixture_size, n_iters):
        al_vec, be_vec, pi_vec = E2.select_initial_component_params(mixture_size)

        tauX_mat = np.zeros((sample_size, mixture_size))
        tauY_mat = np.zeros((sample_size, mixture_size))
        
        for _ in range(n_iters):
            for zz in range(mixture_size):
                tauX_mat[:, zz] = pi_vec[zz] * np.exp(log_dbeta_primitive(XX, al_vec[zz], be_vec[zz]))
                tauY_mat[:, zz] = pi_vec[zz] * np.exp(log_dbeta_primitive(YY, al_vec[zz], be_vec[zz]))
            
            tau_denom_X = np.sum(tauX_mat, axis=1) + 1
            tau_denom_Y = np.sum(tauY_mat, axis=1) + 1
            
            for zz in range(mixture_size):
                tauX_mat[:, zz] /= tau_denom_X
                tauY_mat[:, zz] /= tau_denom_Y
            
            pi_vec = np.sum(tauX_mat + tauY_mat, axis=0) / np.sum(tauX_mat + tauY_mat)
            params = np.array(
                [update_params(i, tauX_mat, tauY_mat, XX, YY, al_vec, be_vec) for i in range(mixture_size)]
                )
            al_vec = params[:, 0]
            be_vec = params[:, 1]
            
        return al_vec, be_vec, pi_vec
        
    @func_timer
    def run(self, n_iters=100, save_data: int = False, verbose: bool = False):
        self.est_param_store = {}
        self.risk_store = {}
        self.sample_store = {}
        
        for this_sample_size in self.list_of_sample_sizes:
            self.risk_store[this_sample_size] = {}
            self.est_param_store[this_sample_size] = {}
            self.sample_store[this_sample_size] = {}
            
            for this_mixture_size in self.list_of_mixture_components:
                XX = E2.sample_from_density(this_sample_size)
                YY = E2.sample_from_lifting_fn(this_sample_size)
                
                al_vec, be_vec, pi_vec = self.optimize_params_for_fixed_mixture_components(
                    XX=XX, 
                    YY=YY, 
                    sample_size=this_sample_size, 
                    mixture_size=this_mixture_size, 
                    n_iters=n_iters
                )
                
                int_X  = quad(E2.integrand_X, 0, 1, args=(al_vec, be_vec, pi_vec), limit=1000)[0]
                int_Y  = quad(E2.integrand_Y, 0, 1, args=(al_vec, be_vec, pi_vec), limit=1000)[0]
                risk = -(int_X + int_Y)
                
                self.risk_store[this_sample_size][this_mixture_size] = risk
                self.est_param_store[this_sample_size][this_mixture_size] = {
                    'alpha': al_vec.tolist(),
                    'beta': be_vec.tolist(),
                    'pi': pi_vec.tolist()
                }
                
                print(f"E1, ss={this_sample_size}, mm={this_mixture_size}, risk={risk}") if verbose else None
                
                if save_data:
                    self.sample_store[this_sample_size][this_mixture_size] = {
                        'XX': XX.tolist(),
                        'YY': YY.tolist()
                    }
    
        print("Experiment completed.")
        self.run_flag = True
        return self
                
        
    def plot_single_result(self, sample_size, mixture_size, ax=None, resolution=100, include_X=True, include_Y=True):
        try:
            data = self.sample_store[sample_size][mixture_size]
        except KeyError:
            raise ValueError(f"Sample size {sample_size} not found.")

        alpha_params = self.est_param_store[sample_size][mixture_size]['alpha']
        beta_params = self.est_param_store[sample_size][mixture_size]['beta']
        pi_params = self.est_param_store[sample_size][mixture_size]['pi']
        
        XX, YY = data
        _xx = np.linspace(0, 1, resolution)
        fitted = fitted_density(_xx, alpha_params, beta_params, pi_params, resolution=resolution)
        
        if ax is None:
            _, ax = plt.subplots()
        if include_X:
            ax.hist(XX, bins=100, density=True, alpha=0.5)
        if include_Y:
            ax.hist(YY, bins=100, density=True, alpha=0.5)
        
        ax.plot(_xx, fitted)
        ax.set_title(f"sample size={len(XX)}, mixture size={len(alpha_params)}")
        return ax
    
    def plot_single_components(self, sample_size, mixture_size, ax=None, resolution=100):

        alpha_params = self.est_param_store[sample_size][mixture_size]['alpha']
        beta_params = self.est_param_store[sample_size][mixture_size]['beta']
        pi_params = self.est_param_store[sample_size][mixture_size]['pi']
        
        _xx = np.linspace(0, 1, resolution)
        if ax is None:
            _, ax = plt.subplots()
        
        for alpha_val, beta_val, pi_val in zip(alpha_params, beta_params, pi_params):
            ax.plot(_xx, pi_val * beta.pdf(_xx, alpha_val, beta_val))
        
        return ax
        
    def plot_results(self, **kwargs):
        if not self.run_flag:
            raise ValueError("Experiment not run yet.")
        
        for this_sample_size in self.list_of_sample_sizes:
            for this_mixture_size in self.list_of_mixture_components:
                self.plot_single_result(this_sample_size, this_mixture_size, **kwargs)
                
    def plot_components(self, **kwargs):
        if not self.run_flag:
            raise ValueError("Experiment not run yet.")
        
        for this_sample_size in self.list_of_sample_sizes:
            for this_mixture_size in self.list_of_mixture_components:
                self.plot_single_components(sample_size=this_sample_size, mixture_size=this_mixture_size, **kwargs)