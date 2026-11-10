import qutip as qt
import numpy as np
from qutip import entropy_vn


class BaseFermiHubbard:
    """Common functionality for all encodings of the double-chain Fermi-Hubbard model.

    Subclasses are expected to:
      - define their own __init__ that accepts encoding-specific parameters
        (e.g. J for the domain-wall encoding), store them on self, then call
        super().__init__(chain_length, gx, gz, e, T, nsteps).
      - override _set_system_size, build_H, _apply_excitation.

    Everything else (operator construction, observables, time evolution, entropy)
    is shared.
    """

    encoding = None  # subclasses set this to a string label

    def __init__(self, chain_length, t, U, e, T, nsteps):
        self.t = t
        self.U = U
        self.e = e

        self._set_system_size(chain_length)

        self.T = T
        self.t_list = np.linspace(0, T, nsteps)

        self.ket0 = qt.basis(2, 0)
        self.ket1 = qt.basis(2, 1)

        self.sx_list, self.sy_list, self.sz_list = self.initialize_operators()
        self.H = self.build_H()
        self.initialize_chain()

    # ---- methods subclasses must implement --------------------------------

    def _set_system_size(self, chain_length):
        raise NotImplementedError

    def build_H(self):
        raise NotImplementedError

    def _apply_excitation(self, occ, chain, site):
        raise NotImplementedError

    # ---- shared functionality ---------------------------------------------

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

    def _resolve(self, chain, site):
        if chain == 'up':
            offset = 0
            assert site < self.chain_length, 'invalid site'
        elif chain == 'down':
            offset = self.chain_length
            assert site + self.chain_length < self.nqubits, 'invalid site'
        else:
            raise ValueError("Invalid chain: select 'up' or 'down'")
        return offset, site + offset
    
    def initialize_chain(self):
        self.branches = [(1.0, [0] * self.nqubits)]

    def place_single_excitation(self, chain, site):
        for _, occ in self.branches:
            self._apply_excitation(occ, chain, site)

    def place_entangled_pair(self, chain1, site1, chain2, site2, phase=+1):
        new_branches = []
        for amp, occ in self.branches:
            occ_a = occ.copy()
            occ_b = occ.copy()
            self._apply_excitation(occ_a, chain1, site1)
            self._apply_excitation(occ_b, chain2, site2)
            new_branches.append((amp, occ_a))
            new_branches.append((amp * phase, occ_b))
        self.branches = new_branches

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
        up_sites = list(range(0, self.chain_length))
        down_sites = list(range(self.chain_length, 2*self.chain_length))

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
            rho_up = state.ptrace(list(range(0, self.chain_length)))
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


class StandardFermiHubbard(BaseFermiHubbard):
    """Direct site encoding: one qubit per site, |1> = occupied."""

    encoding = "st"

    def __init__(self, chain_length, t, U, e, T, nsteps):
        super().__init__(chain_length, t, U, e, T, nsteps)

    def _set_system_size(self, chain_length):
        self.chain_length = chain_length
        self.nqubits = self.chain_length * 2

    def _apply_excitation(self, occ, chain, site):
        _, idx = self._resolve(chain, site)
        occ[idx] = 1

    def build_H(self):
        H = 0

        #Chemical potential
        for i in range(self.nqubits):
            H += self.e * self.sz_list[i]

        # Hopping terms in upper chain
        for i in range(self.chain_length - 1):
            H += self.t * (self.sx_list[i] * self.sx_list[i + 1]
                            + self.sy_list[i] * self.sy_list[i + 1])
            
        #Hopping terms in lower chain    
        for i in range(self.chain_length, self.nqubits - 1):
            H += self.t * (self.sx_list[i] * self.sx_list[i + 1]
                            + self.sy_list[i] * self.sy_list[i + 1])

        #Interaction terms
        for i in range(self.chain_length):
            H += self.U * (self.sz_list[i] * self.sz_list[self.chain_length + i])

        return H


class DomainWallFermiHubbard(BaseFermiHubbard):
    """Domain-wall encoding: occupation encoded as position of a domain wall.

    Requires one extra qubit per chain compared to the standard encoding.
    """

    encoding = "dw"

    def __init__(self, chain_length, t, U, e, T, nsteps, J):
        self.J = J
        super().__init__(chain_length, t, U, e, T, nsteps)

    def _set_system_size(self, chain_length):
        self.chain_length = chain_length + 1
        self.nqubits = self.chain_length * 2

    def _apply_excitation(self, occ, chain, site):
        offset, idx = self._resolve(chain, site)
        for i in range(offset, idx + 1):
            occ[i] ^= 1

    def flip_chain(self, chain):
        for _, occ in self.branches:
            offset, idx = self._resolve(chain, self.chain_length - 1)
            for i in range(offset, idx + 1):
                occ[i] ^= 1

    def build_H(self):
        H = 0

       #Local fields (not related to chemical potential) 
        for i in range(0, self.nqubits):
            H += self.e * (self.sz_list[i])

        #Transverse field in upper chain (skip first one to impose BD)
        for i in range(1, self.chain_length - 1):
            H += self.t * (self.sx_list[i])

        #Transverse field in lower chain (skip last one to impose BD)
        for i in range(self.chain_length + 1, self.nqubits - 1):
            H += self.t * (self.sx_list[i])

        #Domain wall coupling in upper chain (skip last one)
        for i in range(0, self.chain_length - 1):
            H += self.J * self.sz_list[i] * self.sz_list[i + 1]

        #Domain wall coupling in lower chain (skip last one)
        for i in range(self.chain_length, self.nqubits - 1):
            H += self.J  * self.sz_list[i] * self.sz_list[i + 1]

        #On-site interactions (skip last one)
        for i in range(0, self.chain_length - 1):
            H += self.U * (self.sz_list[i] - self.sz_list[i + 1]) \
                        * (self.sz_list[i + self.chain_length] - self.sz_list[i + self.chain_length + 1])
        return H


class HardwareFermiHubbard(BaseFermiHubbard):
    """Skeleton for a third encoding. Fill in the TODOs."""

    encoding = "custom"

    def __init__(self, chain_length, t, U, e, T, nsteps, J, h):
        self.J = J
        self.h = h
        super().__init__(chain_length, t, U, e, T, nsteps)

    def _set_system_size(self, chain_length):
        self.chain_length = chain_length + 1
        # From 0 to chain_length-1 is upper chain,
        # From chain_length to 2*chain_length - 1 is lower chain (to make apply_excitation reusable)
        # From 2*chain_length to 3*chain_length - 2 are auxiliary qubits
        self.nqubits = 3 * self.chain_length - 1

    def _apply_excitation(self, occ, chain, site):
        offset, idx = self._resolve(chain, site)
        for i in range(offset, idx + 1):
            occ[i] ^= 1

    def flip_chain(self, chain):
        for _, occ in self.branches:
            offset, idx = self._resolve(chain, self.chain_length - 1)
            for i in range(offset, idx + 1):
                occ[i] ^= 1

    def initialize_chain(self):
        # Branches describe only the 2*chain_length chain qubits; ancillas are
        # added in |-> at state-construction time in run().
        self.branches = [(1.0, [0] * (2 * self.chain_length))]

    def run(self, pauli_strings=None):
        kets = {0: self.ket0, 1: self.ket1}
        minus = (self.ket0 - self.ket1).unit()
        n_aux = self.chain_length - 1

        psi_chain = sum(amp * qt.tensor([kets[b] for b in occ])
                        for amp, occ in self.branches)
        psi_aux = qt.tensor([minus] * n_aux)
        self.psi0 = qt.tensor(psi_chain, psi_aux).unit()

        sim = qt.sesolve(self.H, self.psi0, self.t_list)

        if pauli_strings is None:
            return sim, None

        measurements = self.calculate_pauli_observables(sim, pauli_strings)
        return sim, measurements

    def build_H(self):
        H = 0

        #Local fields (not related to chemical potential) (not applied to auxiliary qubits)
        for i in range(0, 2*self.chain_length):
            H += self.e * (self.sz_list[i])

        #Transverse field in upper chain (skip first one to impose BD)
        for i in range(1, self.chain_length - 1):
            H += self.t * (self.sx_list[i])

        #Transverse field in lower chain (skip last one to impose BD)
        for i in range(self.chain_length + 1, 2*self.chain_length - 1):
            H += self.t * (self.sx_list[i])

        #Transverse field in auxiliary chain
        for i in range(2*self.chain_length , self.nqubits):
            H += self.h * (self.sx_list[i])

        #Domain wall coupling in upper chain (skip last one)
        for i in range(0, self.chain_length - 1):
            H += self.J * self.sz_list[i] * self.sz_list[i + 1]

        #Domain wall coupling in lower chain (skip last one)
        for i in range(self.chain_length, 2*self.chain_length - 1):
            H += self.J * self.sz_list[i] * self.sz_list[i + 1]

        #On-site interactions (skip last one)
        for i in range(0, self.chain_length - 1):
            H += self.U * (self.sz_list[i] - self.sz_list[i + 1] + \
                           self.sz_list[i + self.chain_length] - self.sz_list[i + self.chain_length + 1])*self.sz_list[i + 2*self.chain_length]
        return H

