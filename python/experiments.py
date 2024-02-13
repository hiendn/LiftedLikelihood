from e1 import E1
from e2 import E2
from utils import *

def main():
    n_iters = 200
    n_replications = 50
    num_sample_points = [2**x for x in range(10, 16)]
    num_mixture_components = [x for x in range(1, 9)]
    
    E1_experiment = E1(num_sample_points, num_mixture_components)
    for i in range(1, n_replications+1):
        print(f"E1. Running replication {i} of {n_replications}")
        E1_experiment.run(n_iters=n_iters, save_data=False)
        save_simulation_outputs(i, E1_experiment.risk_store, 'results', 'e1_results')
        save_simulation_outputs(i, E1_experiment.est_param_store, 'params', 'e1_params')

    E2_experiment = E2(num_sample_points, num_mixture_components)
    for i in range(1, n_replications+1):
        print(f"E2. Running replication {i} of {n_replications}")
        E2_experiment.run(n_iters=n_iters, save_data=False)
        save_simulation_outputs(i, E2_experiment.risk_store, 'results', 'e2_results')
        save_simulation_outputs(i, E2_experiment.est_param_store, 'params', 'e2_params')

if __name__ == '__main__':
    main()
