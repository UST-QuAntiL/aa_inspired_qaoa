import json
import numpy as np

from sympy.logic.boolalg import BooleanFunction
from qiskit.algorithms.minimum_eigen_solvers import VQEResult
from qiskit.circuit import Parameter

### JSON encode for the experiment results

class ResultEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BooleanFunction):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, VQEResult):
            return str(obj.__dict__)
        if isinstance(obj, Parameter):
            return str(obj)
        if isinstance(obj, np.complex128):
            return str(obj)

        
        return json.JSONEncoder.default(self, obj)