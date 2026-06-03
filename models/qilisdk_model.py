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
        self.initialize_chain()

    def initialize_chain(self):
        self.branches = [(1.0, [0] * self.nqubits)]

    def place_single_excitation(self, chain, site):
        if chain == 'up':
            offset = 0
            assert site < self.chain_length, 'invalid site: select value from 0 to chain_length-1'
        elif chain == 'down':
            offset = self.chain_length
            assert site + self.chain_length < self.nqubits, 'invalid site: select value from 0 to chain_length-1'
        else:
            raise ValueError("Invalid chain specified: select 'up' or 'down'")

        idx = site + offset
        for _, occ in self.branches:
            occ[idx] = 1

    def place_entangled_pair(self, site1, site2, phase=+1):
        new_branches = []
        for amp, occ in self.branches:
            occ_a = occ.copy(); occ_a[site1] = 1
            occ_b = occ.copy(); occ_b[site2] = 1
            new_branches.append((amp, occ_a))
            new_branches.append((amp*phase, occ_b))
        self.branches = new_branches

    def build_initial_state(self):
        psi = sum(amp * tensor_prod([basis_state(b, 2) for b in occ])
                  for amp, occ in self.branches)
        return psi.unit() if hasattr(psi, 'unit') else psi

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
        analog_evolution = AnalogEvolution(
            schedule=Schedule.linear(self.H_tot, self.H_tot, total_time=self.T, dt=self.dt),
            initial_state=self.build_initial_state(),
            store_intermediate_results=True,
        )

        backend = QiliSim()
        results = backend.execute(
            analog_evolution,
            Readout().with_expectation(observables=[Z(i) for i in range(self.nqubits)]),
        )

        return np.array(results.get_intermediate_expectation_values())
