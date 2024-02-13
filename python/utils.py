import numpy as np
from scipy.stats import beta
from scipy.special import gammaln
from scipy.optimize import minimize
from datetime import datetime
from functools import wraps as _wraps
from time import perf_counter
import os
import json

def func_timer(fn):
    @_wraps(fn)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        out = fn(*args, **kwargs)
        end = perf_counter()
        print(f'CALLING Function {fn.__name__}. TIME {(end-start):.4f} seconds.')
        return out
    return wrapper

def log_dbeta_primitive(x, shape1, shape2):
    log_numerator = gammaln(shape1 + shape2)
    log_denominator = gammaln(shape1) + gammaln(shape2)
    log_result = log_numerator - log_denominator + (shape1 - 1) * np.log(x) + (shape2 - 1) * np.log(1 - x)
    return log_result

def update_params(index, tauX_mat, tauY_mat, XX, YY, al_vec, be_vec):
    def func(params):
        exp_params = np.exp(params)
        result = -np.sum(
            tauX_mat[:, index] * log_dbeta_primitive(XX, exp_params[0], exp_params[1]) +
            tauY_mat[:, index] * log_dbeta_primitive(YY, exp_params[0], exp_params[1])
        )
        return result

    init_params = np.log([al_vec[index], be_vec[index]])
    bounds = [(np.exp(-5), np.exp(5))] * 2
    res = minimize(func, init_params, bounds=bounds, method='L-BFGS-B')
    return np.exp(res.x)

def fitted_density(xx: np.ndarray, al_vec: np.ndarray | list, be_vec: np.ndarray | list, pi_vec: np.ndarray | list, resolution: int = 100):
    fitted = np.zeros(resolution)
    for this_number in range(resolution):
        for alpha_val, beta_val, pi_val in zip(al_vec, be_vec, pi_vec):
            fitted[this_number] += pi_val * beta.pdf(xx[this_number], alpha_val, beta_val)
    return fitted

def save_simulation_data(xx, simulation_result):
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_simulation_" + str(xx) + ".npz"
    np.savez(filename, simulation_result=simulation_result)
    print("Simulation data saved to:", filename)
    
def save_simulation_data_csv(xx, simulation_result):
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_simulation_" + str(xx) + ".csv"
    np.savetxt(filename, simulation_result, delimiter=",")
    print("Simulation data saved to:", filename)
    
def save_simulation_results_json(simulation_num: int, simulation_results: dict, path_):
    filename = os.path.join(path_, f'{datetime.now().strftime("%Y_%m_%d")}_simulation_{simulation_num}_results.json')
    with open(filename, 'w') as fp:
        json.dump(simulation_results, fp, indent=4)
    print("Simulation results saved to:", filename)
    
def save_simulation_data_json(simulation_num: int, simulation_data: dict, path_):
    filename = os.path.join(path_, f'{datetime.now().strftime("%Y_%m_%d")}_simulation_{simulation_num}_data.json')
    with open(filename, 'w') as fp:
        json.dump(simulation_data, fp, indent=4)
    print("Simulation data saved to:", filename)

def save_simulation_params_json(simulation_num: int, simulation_params: dict, path_):
    filename = os.path.join(path_, f'{datetime.now().strftime("%Y_%m_%d")}_simulation_{simulation_num}_params.json')
    with open(filename, 'w') as fp:
        json.dump(simulation_params, fp, indent=4)
    print("Simulation data params to:", filename)
    
def save_simulation_outputs(simulation_num, simulation_dict: dict, filename: str, path: str = '', verbose: bool = False) -> None:
    output_filename = os.path.join(path, f'{datetime.now().strftime("%Y_%m_%d")}_simulation_{simulation_num}_{filename}.json')
    with open(output_filename, 'w') as fp:
        json.dump(simulation_dict, fp, indent=4)
    print(f"Simulation {filename} saved to: {output_filename}") if verbose else None
