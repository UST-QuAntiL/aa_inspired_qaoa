import mmap
import json
import numpy as np
import ast
import qaoa_3sat as qa

import matplotlib as mpl
import matplotlib.pyplot as plt

from pysat.solvers import Solver


### Convenience functions for JSON data ###

def read_result(resultsfile):
    """yield blocks of results parsed from json"""

    last_index = 0

    with open(resultsfile, 'rb', 0) as file, \
        mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as s:
        while (found_index := s.find(b'}{\n')) != -1:
            #last_index = found_index + 1
            found_index = found_index + 1 #to skip over the closing braket
            val = ''
            val = s.read(found_index-last_index).decode()
            last_index = found_index
            val = " ".join(val.split())
            try:
                yield json.loads(val)
            except: 
                print('exception in generator')
                #print(val)


def sat(cnf_formula):
    """calls sat solver (used for computing classical decision)"""
    with Solver(bootstrap_with=cnf_formula) as m:
        return m.solve()

def get_exp_results(resultsfile):
    """returns experiment results as two sets:
    success_probs: cl_to_var_ratio -> success_prob 
    sat_probs: cl_to_var_ratio -> satisfiability property of the input 
    """
    n = 0 #will be read from file - should be the same for all entries
    success_probs = dict()
    sat_probs = dict()
    for res in read_result(resultsfile):
        n = res['n']
        m = res['m']
        results = res['results'] 
        sum_correct = 0
        sum_sat = 0
        for single_exp in results:
            qaoa_decision = single_exp[3]
            classical_decision = sat(single_exp[0])
            if qaoa_decision == classical_decision:
                sum_correct += 1
            if classical_decision:
                sum_sat += 1
        success_probs[np.round(m/n, 4)] = sum_correct/len(results)
        sat_probs[np.round(m/n, 4)] = sum_sat/len(results)
    
    return success_probs, sat_probs


def get_exp_results_th(resultsfile, threshold):
    """returns experiment results as two sets:
    success_probs: cl_to_var_ratio -> success_prob 
    sat_probs: cl_to_var_ratio -> satisfiability property of the input 
    
    Assumes QAOA deemed the input satisfiable iff the ratio of 
    measured unsatisfying assignments is below the threshold.
    Uses the cost function value for this purpose since it encodes 
    the ratio of unsatisfied assignments (using the binary cost function).
    """
    n = 0 #will be read from file - should be the same for all entries
    success_probs = dict()
    sat_probs = dict()
    for res in read_result(resultsfile): # one json object for each m
        n = res['n']
        m = res['m']
        results = res['results'] #= all results for this m
        sum_correct = 0
        sum_sat = 0
        for single_exp in results:
            cf_val = ast.literal_eval(single_exp[1].replace("array", ""))['fun'] # extracts cost function value ("fun") from optimizer result
            qaoa_decision = cf_val <= threshold
            classical_decision = sat(single_exp[0])
            if qaoa_decision == classical_decision:
                sum_correct += 1
            if classical_decision:
                sum_sat += 1
        success_probs[np.round(m/n, 4)] = sum_correct/len(results)
        sat_probs[np.round(m/n, 4)] = sum_sat/len(results)
    
    return success_probs, sat_probs


### Plot formatting

def plot_dicts(*dicts, maxy = 1.05, miny=0, labels=None, sat_probs=None, start_at=15, fontsize=10):
    """plots a dict that maps x to y values
    Allows plotting multiple dicts at once and assigns given labels"""
    fig = plt.figure()

    plt.xlabel(r'clause-to-variable ratio $\alpha$', fontsize=fontsize)
    plt.ylabel("success probability", fontsize=fontsize)

    # colormap boundaries
    cmap = mpl.cm.get_cmap('copper_r')
    cmap.set_gamma(2.0)
    norm = mpl.colors.BoundaryNorm([key-0.5 for key in range(0, len(dicts)+1)], cmap.N, extend='both')
    markers = ["o", "v", "*", "s", "P", "8", "^", "1"]

    # Plot given satisfiability probabilities in dottet line
    if sat_probs != None:
        dict_s = sorted(sat_probs.items()) 
        x, y = zip(*dict_s) 
        plt.ylim([miny, maxy])
        plt.plot(x[start_at:],y[start_at:], label='SAT probability', linestyle="dotted", color="grey")

    # Plot given dictionaries using colormap
    i = 0
    for rdict in dicts:
        dict_s = sorted(rdict.items()) 
        x, y = zip(*dict_s) 
        plt.ylim([miny, maxy])
        if labels != None:
            plt.plot(x[start_at:],y[start_at:], label=labels[i], marker=markers[i], linewidth=0.5, color=cmap(norm(i)), markersize=4) 
        else:
            plt.plot(x[start_at:],y[start_at:], marker=markers[i], color=cmap(norm(i)), markersize=6)
        i += 1
        
    if labels != None:
        plt.legend(fontsize=fontsize)
    return fig