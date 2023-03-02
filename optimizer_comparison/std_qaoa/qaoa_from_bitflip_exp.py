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
from qiskit.opflow import PauliSumOp, Z, I, X, One
import random

from multiprocessing import Pool, Lock
import multiprocessing

import random_3sat as qa
import std_qaoa.qaoa_from_bitflip as qbf
from qiskit import Aer, transpile
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SPSA
from util import ResultEncoder

def append_tensor(clause_op, append):
    if clause_op == None:
        return append
    else:
        return clause_op ^ append

def get_clause_hamiltonian_min(clause, atoms):
    identity = None 
    for _ in atoms:
        identity = append_tensor(identity, I)
    
    clause_atoms = [abs(lit) for lit in clause]
    clause_op = None
    for atom in atoms:
        if atom in clause_atoms:
            # polarity
            if atom in clause: # it is positive in the clause
                clause_op = append_tensor(clause_op, (1/2)*(I+Z))
            else: # it is negative in the clause
                clause_op = append_tensor(clause_op, (1/2)*(I-Z))
        else: # it is not in the clause
            clause_op = append_tensor(clause_op, I)
            
    return clause_op # max3sat minimization version does not subtract from identity


def cost_operator_maxsat(pysat_formula, atoms):
    # using cost function formula found in e.g. Zhang et al.
    # Ex. formula: [[-2, 1, -4], [-3, -4, -5], [3, 4, -1], [-2, -3, 1]]
    #print("Formula: ", pysat_formula)

    cf = 0 

    for clause in pysat_formula:
        cf = cf + get_clause_hamiltonian_min(clause, atoms)
    
    return cf

output_lock = Lock()

def get_obj_value_fn(formula):
    def obj_value_fn(bitvec):
        # LSB of bitvec ist uppermost qubit is LSB of cost op = highest most variable in formula
        sat_clauses = 0
        for clause in formula:
            for lit in clause:
                if lit < 0:
                    if bitvec[abs(lit)-1] == '0':
                        sat_clauses += 1
                        break
                if lit > 0:
                    if bitvec[abs(lit)-1] == '1':
                        sat_clauses += 1
                        break

        return len(formula) - sat_clauses
    return obj_value_fn

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
        cost_op = (cost_operator_maxsat(formula, symbols)).reduce()

        best_fval_for_formula = 1000
        best_save_data_for_formula = None

        # retry and save best result of all retries
        for j in range(0, num_retries):

            qaoa = qbf.QAOAbf(
                cost_op=cost_op,
                obj_value_fn=get_obj_value_fn(formula)
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

        results.append(best_save_data_for_formula)

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