import numpy as np
from qilisdk.analog import Schedule, X, Z, Y
from qilisdk.core import QTensor
from qilisdk.core.interpolator import Interpolation
from qilisdk.backends import QiliSim, CudaBackend
from qilisdk.functionals import AnalogEvolution

# Define total time and timestep
T = 10.0
dt = 0.5
nqubits = 1

# Define Hamiltonians
Hx = -sum(X(i) for i in range(nqubits))
Hz = sum(Z(i) for i in range(nqubits))

# Create the AnalogEvolution functional
analog_evolution = AnalogEvolution(
    schedule=Schedule.linear(Hx, Hz, total_time=T, dt=dt),
    initial_state=QTensor.uniform(nqubits),
    store_intermediate_results=False,
)