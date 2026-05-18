import qutip as qt
import numpy as np


class QutipFermiHubbard:
    def __init__(self, chain_length, gx, gz, e, T, nsteps):
        self.chain_length = chain_length
        self.nqubits = chain_length*2
        self.gx = gx
        self.gz = gz
        self.e = e
        self.T = T
        self.t_list = np.linspace(0, T, nsteps)

        self.ket0 = qt.basis(2, 0)
        self.ket1 = qt.basis(2, 1)

        self.sx_list, self.sy_list, self.sz_list = self.initialize_operators()
        self.H = self.build_H()
        self.initialize_chain()

    def initialize_operators(self):
        sx_list, sy_list, sz_list = [], [], []
        sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()
        for i in range(self.nqubits):
            op_list = [qt.qeye(2)] * self.nqubits
            op_list[i] = sx
            sx_list.append(qt.tensor(op_list))
            op_list[i] = sy
            sy_list.append(qt.tensor(op_list))
            op_list[i] = sz
            sz_list.append(qt.tensor(op_list))

        return sx_list, sy_list, sz_list
    
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

    def build_H(self):
        
        H = 0

        for i in range(self.nqubits):
            H += (self.e/2)*self.sz_list[i]

        for i in range(self.chain_length-1):
            H += self.gx*(self.sx_list[i]*self.sx_list[i+1] + self.sy_list[i]*self.sy_list[i+1])

        for i in range(self.chain_length, self.nqubits-1):
            H += self.gx*(self.sx_list[i]*self.sx_list[i+1] + self.sy_list[i]*self.sy_list[i+1])

        for i in range(self.chain_length):
            H += self.gz*(self.sz_list[i]*self.sz_list[self.chain_length+i])

        return H

    def calculate_observable_along_chain(self, state_evolution, observable):
        return np.array([[qt.expect(op, state)
                            for op in observable]
                           for state in state_evolution.states])

    def run(self):
        kets = {0: self.ket0, 1: self.ket1}
        psi = sum(amp * qt.tensor([kets[b] for b in occ])
                  for amp, occ in self.branches)
        self.psi0 = psi.unit()

        sim = qt.sesolve(self.H, self.psi0, self.t_list)

        magn_z = self.calculate_observable_along_chain(sim, self.sz_list)
        return magn_z
