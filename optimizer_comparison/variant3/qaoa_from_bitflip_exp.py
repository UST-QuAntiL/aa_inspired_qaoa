import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import itertools as it
from functools import partial
import numpy as np
from sympy.logic.boolalg import BooleanFunction
from qiskit.algorithms.minimum_eigen_solvers import VQEResult
from qiskit.circuit import Parameter
import json

from multiprocessing import Pool, Lock
import multiprocessing

import random_3sat as qa
import variant3.qaoa_from_bitflip as qbf
from qiskit import Aer, transpile
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SPSA
from util import ResultEncoder


output_lock = Lock()


def run_step(m, n, number_of_formulas, p, save_dir, warm_started_layers=0, num_retries=1):
    """
    Runs one step of the experiment = evaluates 'number_of_fomulas' formulas for specific 
    number fo clauses 'm' and appends results to 'save_dir/results_n.json'.
    
    if warm_started_layers > 0, QAOA will be first run for 
    the reduced number of layers supplied in this variable and the 
    resulting parameters will be used for initialisiation of the remaining
    optimization step - the remaining parameters will be sampled 
    uniformly random in [0,0.1] to initialize"""
    global output_lock

    pid = os.getpid()
    batch = m # m is used to store the batch id
    m=45

    print("solving n=%d, m=%d, runs=%d, p=%d, p'=%d" % (n,m, number_of_formulas, p, warm_started_layers))
    sys.stdout.flush()

    symbols = [i+1 for i in range(n)]

    results = []

    input_formulas = np.load("input_formulas.npy")

    for i in range(0, number_of_formulas):
        
        findex = batch*number_of_formulas + i
        formula = input_formulas[findex]
        qiskit_formula = qbf.get_classical_fn(formula, n)

        best_fval_for_formula = 1000
        best_save_data_for_formula = None

        # retry and save best result of all retries
        for j in range(0, num_retries):

            qaoa = qbf.QAOAbf(
                cost_circuit=qbf.get_cost_circuit(qiskit_formula),
                mixer=qbf.grover_mixer(n),
                obj_value_fn=qbf.get_objective_fn(qiskit_formula)
                )
            initial_parameters=None
            if warm_started_layers > 0:
                ws_result, cts = qaoa.run(warm_started_layers, shots=100, backend=Aer.get_backend('aer_simulator'), optimizer=SPSA())
                initial_parameters = [*ws_result.x[:warm_started_layers],
                            *np.random.uniform(0,0.1,[p-warm_started_layers]),
                            *ws_result.x[warm_started_layers:],
                            *np.random.uniform(0,0.1,[p-warm_started_layers])]
                    

            opt_result, cts, best_fval, best_params = qaoa.run(
                p, shots=100, backend=Aer.get_backend('aer_simulator'), optimizer=SPSA(), 
                initial_parameters=initial_parameters
                )
            

            save_data = (
                formula,
                str(opt_result),
                cts, 
                opt_result.fun < 1, # qaoa decision
                best_fval
            )

            if best_fval < best_fval_for_formula:
                best_fval_for_formula = best_fval
                best_save_data_for_formula = save_data

        results.append(save_data)

    # save it
    with output_lock:
        with open(save_dir + "/results_" + str(n) + "_" + str(pid) + ".json", "a") as write:
            json.dump(
                {
                    "n":n,
                    "m":m,
                    "results":results
                }, write, cls=ResultEncoder, indent=4) 
            write.flush()


def run_and_save(n, p, ratio_min, ratio_max, stepsize, num_per_step, save_dir):
    run_and_save(n, p, ratio_min, ratio_max, stepsize, num_per_step, 0, save_dir)

def run_and_save(n, p, ratio_min, ratio_max, stepsize, num_per_step, warm_started_layers, save_dir):
    m_min = int(ratio_min * n)
    m_max = int(ratio_max * n)
    steps = np.arange(m_min, m_max, stepsize)

    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        print("Output directory exists. Will not start unless directory is deleted to prevent overwriting data.")
        return
    
    with Pool(5) as pool:
        pool.map(partial(
            run_step, 
            n=n,
            save_dir=save_dir,
            p=p,
            number_of_formulas=num_per_step,
            warm_started_layers=warm_started_layers), 
            list(steps))

def run_and_save_fixed_m(n, p, m, num_per_step, save_dir, num_retries=1, num_proc=5):
    steps = [i for i in range(0, num_proc)]
    number_of_formulas = int((num_per_step/num_proc) + 1)
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        print("Output directory exists. Will not start unless directory is deleted to prevent overwriting data.")
        return
    
    with Pool(num_proc) as pool:
        pool.map(partial(
            run_step, 
            n=n,
            save_dir=save_dir,
            p=p,
            number_of_formulas=number_of_formulas,
            warm_started_layers=0,
            num_retries=num_retries), 
            list(steps))