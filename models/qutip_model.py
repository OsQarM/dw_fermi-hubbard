import qutip as qt
import numpy as np
from qutip import entropy_vn


class QutipFermiHubbard:
    def __init__(self, chain_length, gx, gz, e, T, nsteps, J = None, mode = "st"):

        self.mode = mode
        if mode == "st":
            self.chain_length = chain_length
        elif mode == "dw":
            self.chain_length = chain_length+1

        self.nqubits = self.chain_length*2
        self.gx = gx
        self.gz = gz
        self.e = e
        self.J = J
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
        #Single branch, all qubits down, weight 1
        self.branches = [(1.0, [0] * self.nqubits)]

    def _resolve(self, chain, site):
        #Add offset to index if applying operation to down chain
        #Also check that indices are within bounds
        if chain == 'up':
            offset = 0
            assert site < self.chain_length, 'invalid site'
        elif chain == 'down':
            offset = self.chain_length
            assert site + self.chain_length < self.nqubits, 'invalid site'
        else:
            raise ValueError("Invalid chain: select 'up' or 'down'")
        return offset, site + offset

    def _apply_excitation(self, occ, chain, site):
        #Make checks
        offset, idx = self._resolve(chain, site)
        #If direct encoding only flip index
        if self.mode == 'st':
            occ[idx] = 1
        #If dw encoding, flip all spins up to index (included)
        elif self.mode == 'dw':
            for i in range(offset, idx + 1):
                occ[i] ^= 1

    def place_single_excitation(self, chain, site):
        #Apply to all branches (important when entangled pairs present)
        for _, occ in self.branches:
            self._apply_excitation(occ, chain, site)

    def place_entangled_pair(self, chain1, site1, chain2, site2, phase=+1):
        new_branches = []
        #Duplicate branches every time we entangle 2 spins
        for amp, occ in self.branches:
            occ_a = occ.copy()
            occ_b = occ.copy()
            self._apply_excitation(occ_a, chain1, site1)
            self._apply_excitation(occ_b, chain2, site2)
            new_branches.append((amp, occ_a))
            #Optional, make antisymetric entanglement with phase=-1
            new_branches.append((amp * phase, occ_b))
        self.branches = new_branches

    def flip_chain(self, chain):
        #Used for changing parity of a chain
        for _, occ in self.branches:
            offset, idx = self._resolve(chain, self.chain_length-1)
            for i in range(offset, idx+1):
                occ[i] ^= 1

    def build_H(self):
        if self.mode == "st":
            return self.build_st_H()
        elif self.mode == "dw":
            return self.build_dw_H()
        
    def build_st_H(self):
        H = 0

        #Chemical potential
        for i in range(self.nqubits):
            H += (self.e/2)*self.sz_list[i]

        #Hopping terms chain up
        for i in range(self.chain_length-1):
            H += self.gx*(self.sx_list[i]*self.sx_list[i+1] + self.sy_list[i]*self.sy_list[i+1])

        #Hopping terms chain down
        for i in range(self.chain_length, self.nqubits-1):
            H += self.gx*(self.sx_list[i]*self.sx_list[i+1] + self.sy_list[i]*self.sy_list[i+1])

        #Interaction terms
        for i in range(self.chain_length):
            H += self.gz*(self.sz_list[i]*self.sz_list[self.chain_length+i])

        return H
    
    def build_dw_H(self):
        H = 0

        #Hopping terms
        #chain up
        for i in range(1, self.chain_length-1): #skip first and last one
            H += self.gx*(self.sx_list[i])

        #chain down
        for i in range(self.chain_length + 1, self.nqubits-1):
            H += self.gx*(self.sx_list[i])

        #Domain wall interactions
        #chain up
        for i in range(0, self.chain_length-1): #skip last one
            H += (self.J - self.e/2)*self.sz_list[i]*self.sz_list[i+1]

        #chain down
        for i in range(self.chain_length, self.nqubits-1):
            H += (self.J - self.e/2)*self.sz_list[i]*self.sz_list[i+1]

        #On-site interacctions
        for i in range(0, self.chain_length-1):
            H += self.gz/4*(self.sz_list[i] - self.sz_list[i+1])*(self.sz_list[i+self.chain_length] - self.sz_list[i+self.chain_length+1])

        return H
    

    def calculate_observable_along_chain(self, state_evolution, observable):
        return np.array([[qt.expect(op, state)
                            for op in observable]
                           for state in state_evolution.states])

    def _single_pauli_op(self, letter, site):
        if letter == 'X':
            return self.sx_list[site]
        if letter == 'Y':
            return self.sy_list[site]
        if letter == 'Z':
            return self.sz_list[site]
        if letter == 'I':
            return qt.tensor([qt.qeye(2)] * self.nqubits)
        raise ValueError(f"Unsupported Pauli letter: {letter}")

    def build_pauli_observables(self, pauli_strings):
        """Build observables for a list of Pauli strings (e.g. ['Z', 'X', 'ZZ']).

        For a k-body string, the operator is placed on k consecutive sites that
        stay entirely within a single chain (no crossing between up and down).
        Returns a dict mapping each string to a list of (start_site, op) tuples.
        """
        up_sites = list(range(0, self.chain_length))
        down_sites = list(range(self.chain_length, self.nqubits))

        observables = {}
        for pstr in pauli_strings:
            k = len(pstr)
            ops = []
            for chain_sites in (up_sites, down_sites):
                for start in range(len(chain_sites) - k + 1):
                    sites = chain_sites[start:start + k]
                    op = self._single_pauli_op(pstr[0], sites[0])
                    for letter, site in zip(pstr[1:], sites[1:]):
                        op = op * self._single_pauli_op(letter, site)
                    ops.append((sites[0], op))
            observables[pstr] = ops
        return observables

    def calculate_pauli_observables(self, state_evolution, pauli_strings):
        observables = self.build_pauli_observables(pauli_strings)
        return {
            pstr: np.array([[qt.expect(op, state) for _, op in ops]
                            for state in state_evolution.states])
            for pstr, ops in observables.items()
        }
    
    def calculate_von_neumann_entropy(self, state_evolution):
        entropy = np.zeros(len(state_evolution))
        for i, state in enumerate(state_evolution):
            rho_up = state.ptrace(list(range(0,self.chain_length)))
            entropy[i] = entropy_vn(rho_up)

        return entropy

    def run(self, pauli_strings=None):
        kets = {0: self.ket0, 1: self.ket1}
        psi = sum(amp * qt.tensor([kets[b] for b in occ])
                  for amp, occ in self.branches)
        self.psi0 = psi.unit()

        sim = qt.sesolve(self.H, self.psi0, self.t_list)

        if pauli_strings is None:
            return sim, None

        measurements = self.calculate_pauli_observables(sim, pauli_strings)
        return sim, measurements
