import numpy as np
import random

### Helpers for generating 3sat instances

def generate_clause_pysat(symbols):
    formula = random.sample(symbols, 3)
    
    polarity = np.binary_repr(random.randint(0,7), 3)
    
    for i, pol in enumerate(polarity):
        if pol == '1':
            formula[i] *= -1
    
    return formula

def gen_random_3sat_pysat(symbols, num_clauses):
    """generates 3sat cnf formulas in the format supported by pysat
    = list of clauses - each clause is a list with 3 elements.
    each element (= literal) is either +k or -k depending on polarity"""
    formula = [generate_clause_pysat(symbols) for i in range(0, num_clauses)]
    
    return formula
