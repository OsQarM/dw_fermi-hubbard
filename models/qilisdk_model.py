import numpy as np
from qilisdk.analog import Schedule, X, Z, Y
from qilisdk.core import QTensor, tensor_prod, basis_state
from qilisdk.backends import QiliSim
from qilisdk.functionals import AnalogEvolution
from qilisdk.readout import Readout


class QilisdkFermiHubbard:
    def __init__(self, chain_length, gx, gz, e, T, dt):
        self.chain_length = chain_length
        self.nqubits = chain_length*2
        self.gx = gx
        self.gz = gz
        self.e = e
        self.T = T
        self.dt = dt

        self.H_tot = self.build_H()

    def build_H(self):
        chain_length = self.chain_length
        nqubits = self.nqubits
        gx, gz, e = self.gx, self.gz, self.e

        Hz         = sum((e/2)*Z(i) for i in range(nqubits))
        H_hop_up   = sum(gx*(X(i)*X(i+1)+Y(i)*Y(i+1)) for i in range(chain_length-1))
        H_hop_down = sum(gx*(X(i)*X(i+1)+Y(i)*Y(i+1)) for i in range(chain_length, nqubits-1))
        H_int      = sum(gz*Z(i)*Z(chain_length +i) for i in range(chain_length))

        return Hz + H_hop_up + H_hop_down + H_int

    def run(self):
        single_qubit_states = [basis_state(1, 2)] + [basis_state(0, 2)]*4 + [basis_state(0, 2)]*4 + [basis_state(1, 2)]

        analog_evolution = AnalogEvolution(
            schedule=Schedule.linear(self.H_tot, self.H_tot, total_time=self.T, dt=self.dt),
            initial_state=tensor_prod(single_qubit_states),
            store_intermediate_results=True,
        )

        backend = QiliSim()
        results = backend.execute(
            analog_evolution,
            Readout().with_expectation(observables=[Z(i) for i in range(self.nqubits)]),
        )

        return np.array(results.get_intermediate_expectation_values())
