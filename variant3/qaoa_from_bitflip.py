from qiskit import Aer, transpile
from qiskit.circuit.library import PhaseOracle

import numpy as np

from tweedledum.bool_function_compiler import bitflip_circuit, BoolFunction
from tweedledum.qiskit.converters import to_qiskit
from qiskit.converters import dag_to_circuit
from qiskit.circuit import classical_function,  Int1
from qiskit.circuit.classicalfunction import ClassicalFunction
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit.circuit import Parameter

from qiskit.algorithms.minimum_eigen_solvers import QAOA
from typing import List, Callable, Optional, Union
from qiskit.algorithms.optimizers import Optimizer
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.providers import Backend
from qiskit.algorithms.optimizers import COBYLA
from qiskit import transpile, Aer, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import PhaseGate
import numpy as np


# Wrapper for QAOA circuit and execution 

class QAOAbf():
    def __init__(
        self,
        cost_circuit,
        mixer,
        obj_value_fn,
        init_circuit = None
    ):
        self.cost_circuit = cost_circuit
        self.mixer_circuit = mixer 
        self.obj_value_fn = obj_value_fn
        self.init_circuit = init_circuit

    def obj_fn(self, counts):
        """Computes average of objective function on measurements.
        combined objective function = mean of individual obj fns"""
        sum_obj = 0
        num_entries = 0
        for bitvec, count in counts.items():
            obj = self.obj_value_fn(bitvec)
            sum_obj += obj * count 
            num_entries += count 
        
        return float(sum_obj)/float(num_entries)

    def build_circuit(self, p):
        """Builds Parametrized QAOA ansatz circuit with p layers.
        Uses cost_circuit and mixer_circuit as given in constructor.
        Variant 2 (and 3) fixes the phase-separator parameter"""
        # create a new set of 2p parameters (gammas and betas) that will replace the parameters in each layer
        betas = [Parameter("B%d" % i) for i in range(0, p)]

        # Overall circuit
        overall_num_qubits = self.cost_circuit.num_qubits
        qaoa_qubits = QuantumRegister(self.mixer_circuit.num_qubits, "q")

        if self.cost_circuit.num_qubits <= self.mixer_circuit.num_qubits:
            print("expected cf to have more qubits!!!")
            print(self.cost_circuit)
            print("Mixer qubits: ", self.mixer_circuit.num_qubits)

        ancilla_qubits = QuantumRegister(self.cost_circuit.num_qubits - self.mixer_circuit.num_qubits, "a")
        measure_bits = ClassicalRegister(self.mixer_circuit.num_qubits, "m")
        circ = QuantumCircuit(qaoa_qubits, ancilla_qubits, measure_bits)

        if self.init_circuit == None: # 'Normal' init circuit (full superposition)
            circ.h(qaoa_qubits)
        else:
            circ.append(self.init_circuit, qaoa_qubits)

        # assumes len(gammas) = len(betas)
        for i in range(0, p):
            # cost 
            circ.append(self.cost_circuit, [*qaoa_qubits, *ancilla_qubits])
            # mixer 
            circ.append(self.mixer_circuit.assign_parameters([betas[i]], inplace=False), qaoa_qubits)

        circ.measure(qaoa_qubits, measure_bits)

        return circ

    def get_executor(self, p, backend, shots):
        """Returns executor function that can be used as input for optimizers."""
        # cf and mixer should each be parametrized
        circuit = self.build_circuit(p)

        tcirc = transpile(circuit, backend)

        def execute(paramlist):
            # transpiled circ with params
            # in the grover test version, all gamma parameters are already set, so they will not be part of the params 
            pcirc = tcirc.bind_parameters(paramlist)

            result = backend.run(pcirc, shots=shots).result()
            cts = result.get_counts(pcirc)

            obj_val = self.obj_fn(cts)

            return obj_val
            
        return execute

    def get_outputs(self, p, parameters, shots, backend):
        """Returns list of output for given parameters"""
        circuit = self.build_circuit(p)
        tcirc = transpile(circuit, backend)
        pcirc = tcirc.bind_parameters(parameters)

        result = backend.run(pcirc, shots=shots).result()
        return result.get_counts(pcirc)

    def run(self, p=1, shots=100, backend=None, optimizer=COBYLA(), initial_parameters=None):
        """Starts optimization loop"""
        execfn = self.get_executor(p, backend, shots)

        if initial_parameters == None:
            initial_parameters = [np.random.random_sample()*np.pi for i in range(0,p)]

        result = optimizer.minimize(fun=execfn, x0=initial_parameters)
        return result, self.get_outputs(p, result.x, shots, backend)


def standard_mixer(n):
    """Returns transverse field mixer for QAOA as parametrized quantum circuit"""
    beta = Parameter("$\\beta$")
    reg = QuantumRegister(n)
    circ = QuantumCircuit(reg)
    
    for i in range(0, n):
        circ.rx(2 * beta, i)
        
    return circ

def grover_mixer(n):
    """Returns the grover mixer for QAOA as parametrized quantum circuit"""
    beta = Parameter("$\\beta$")
    reg = QuantumRegister(n)
    circ = QuantumCircuit(reg)

    circ.h(reg)

    # I + e^(-i*beta)|0><0|
    circ.x(reg)

    mcp = PhaseGate(-beta).control(n-1)
    circ.append(mcp, reg)

    circ.x(reg)

    circ.h(reg)

    return circ

def get_cost_circuit(input_formula):
    """Builds phase separation oracle using bitflip oracle. The bitflip oracle is obtained using qiskit circuit synthesis (= tweedledum)."""
    bcirc = input_formula.synth() # synth method accepts other tweedledum synths
    # Qubit order: parameters for input_formula are assigned from top to bottom

    cost_circ = QuantumCircuit(bcirc.num_qubits)

    # bitflip oracle that has exactly the inverse result
    cost_circ.x(bcirc.num_qubits - 1)
    cost_circ.append(bcirc, range(0, cost_circ.num_qubits))

    # in the grover version this is just set to e^-i*pi = -1 phase
    cost_circ.p((-1)*np.pi, cost_circ.num_qubits - 1) # phase gate on the result qubit

    # uncompute bit flip
    cost_circ.append(bcirc, range(0, bcirc.num_qubits))
    cost_circ.x(cost_circ.num_qubits - 1)

    return cost_circ

def get_objective_fn(input_formula):
    """Returns the cost function for one output: str -> 1 if not satisfied, str -> 0 if satisfied.
    The input strings are ordered lsb = top qubit and there might be more qubits past the msb (= anillas)"""
    num_input_qubits = len(input_formula.args)
    def cf(bitvec):
        # For the tweedledum simulation in the background, the input has to be converted to 
        # a list of booleans 

        # the last 'num_input_qubits' qubits are the actual input bitvec (= the topmost qubits in the circuit result)
        inputs_str = bitvec[-num_input_qubits:]
        
        # These are then reversed and converted to a list since the simulate function takes the first argument (= lsb) first
        inputs = [c == '1' for c in reversed(inputs_str)]
        
        # evaluate
        output = input_formula.simulate(inputs) # list containing only one output

        return 0 if output[0] else 1 # return 0 for satisfying assignments and 1 for unsat assignments

    return cf



# Formula input generation from PYSAT

def lit_to_code(lit):
    if lit < 0:
        return 'not x' + str(np.abs(lit))
    else:
        return 'x' + str(np.abs(lit))

def clause_to_code(clause):
    # first var 
    cl_code = lit_to_code(clause[0])
    for lit in clause[1:]:
        cl_code += ' or ' + lit_to_code(lit)
    return '(' + cl_code + ')'

def get_classical_fn(pysat_formula, n):
    """Utility function that transforms a pysat CNF (= list of list) formula to a classial function object for qiskit""" 

    symbols = ["x%d" % (i+1) for i in range(0, n)] # using i+1 to reflect naming for pysat formulas
    
    # function signature 
    # boolfunc(a: Int1, b: Int1, c:Int1) -> Int1:
    sig = 'def boolfunc(' + ''.join(["%s: Int1," % symb for symb in symbols[:-1]]) + ("%s: Int1" % symbols[-1]) + ') -> Int1:'
    body = clause_to_code(pysat_formula[0])
    for clause in pysat_formula[1:]:
        body += ' and ' + clause_to_code(clause)
    
    body = '    return ' + body
    
    cl = ClassicalFunction(sig + '\n' + body)
    return cl