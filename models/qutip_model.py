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

        self.sx_list, self.sy_list, self.sz_list = self.initialize_operators(self.nqubits)
        self.H = self.build_H(self.nqubits, self.chain_length, self.gx, self.gz, self.e)

    def initialize_operators(self, nqubits):
        sx_list, sy_list, sz_list = [], [], []
        sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()
        for i in range(nqubits):
            op_list = [qt.qeye(2)] * nqubits
            op_list[i] = sx
            sx_list.append(qt.tensor(op_list))
            op_list[i] = sy
            sy_list.append(qt.tensor(op_list))
            op_list[i] = sz
            sz_list.append(qt.tensor(op_list))

        return sx_list, sy_list, sz_list

    def build_H(self, nqubits, chain_length, gx, gz, e):
        sx_list, sy_list, sz_list = self.sx_list, self.sy_list, self.sz_list
        H = 0

        for i in range(nqubits):
            H += (e/2)*sz_list[i]

        for i in range(chain_length-1):
            H += gx*(sx_list[i]*sx_list[i+1] + sy_list[i]*sy_list[i+1])

        for i in range(chain_length, nqubits-1):
            H += gx*(sx_list[i]*sx_list[i+1] + sy_list[i]*sy_list[i+1])

        for i in range(chain_length):
            H += gz*(sz_list[i]*sz_list[chain_length+i])

        return H

    def calculate_observable_along_chain(self, state_evolution, observable):
        return np.array([[qt.expect(op, state)
                            for op in observable]
                           for state in state_evolution.states])

    def run(self):
        psi_0_list = [self.ket1] + [self.ket0]*(self.nqubits-2) + [self.ket1]
        psi0 = qt.tensor(psi_0_list)

        sim = qt.sesolve(self.H, psi0, self.t_list)

        magn_z = self.calculate_observable_along_chain(sim, self.sz_list)
        return magn_z
