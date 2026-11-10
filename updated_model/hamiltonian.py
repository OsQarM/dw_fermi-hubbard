import numpy as np

from functools import reduce  
import operator
import warnings

from qilisdk.analog import Schedule, Hamiltonian, X, Z, Y, I
from qilisdk.core import QTensor, tensor_prod, basis_state
from qilisdk.backends import QiliSim
from qilisdk.functionals import AnalogEvolution
from qilisdk.readout import Readout


class SpinHamiltonian:

    def __init__(self, h_type, L, b, t, U, mu, eta, h, J=None):
        self.length = L
        self.h_type = h_type
        self.VALID_TYPES = ("spin", "exact_dw", "approx_dw")

        self.b   = b
        self.t   = t
        self.U   = U
        self.mu  = mu 
        self.eta =  eta
        self.h   = h 
        self.J   = J
    
        self.nqubits = self.define_nqubits()

        self.check_initialization()
        self.initialize_chain()
        self.H = self.build_H()


    def define_nqubits(self):
        self.N = self.length if self.h_type == "spin" else self.length + 1   # qubits per chain
        return 2 * self.N

    def initialize_chain(self):
        """Empty system. DW: no walls, so every qubit of a chain equals its left edge:
        up chain Z = +1 (bit 0), down chain Z = b (bit 1 if b = -1). JW: all qubits empty."""
        N = self.nqubits // 2
        down_bit = 1 if (self.h_type != "spin" and self.b == -1) else 0
        self.branches = [(1.0, [0] * N + [down_bit] * N)]

    def _qubit(self, chain, k):
        if chain not in ("up", "down"):
            raise ValueError("chain must be 'up' or 'down'")
        return k + (self.nqubits // 2 if chain == "down" else 0)

    def _apply_excitation(self, occ, chain, site):
        """Add a particle on fermion site `site` (0..L-1) of `chain`, in place."""
        if not 0 <= site < self.length:
            raise ValueError(f"site must be in 0..{self.length - 1}, got {site}")
        if self.h_type == "spin":
            q = self._qubit(chain, site)
            if occ[q]:
                raise ValueError(f"site {site} ({chain}) is already occupied")
            occ[q] = 1
        else:
            # particle on site i = wall on bond (i, i+1): flip qubits i+1..L; the left edge (b) never changes
            if occ[self._qubit(chain, site)] != occ[self._qubit(chain, site + 1)]:
                raise ValueError(f"site {site} ({chain}) is already occupied")
            for k in range(site + 1, self.N):
                occ[self._qubit(chain, k)] ^= 1

    def place_single_excitation(self, chain, site):
        for _, occ in self.branches:
            self._apply_excitation(occ, chain, site)

    def place_entangled_pair(self, chain1, site1, chain2, site2, phase=+1):
        new_branches = []
        for amp, occ in self.branches:
            occ_a, occ_b = occ.copy(), occ.copy()
            self._apply_excitation(occ_a, chain1, site1)
            self._apply_excitation(occ_b, chain2, site2)
            new_branches += [(amp, occ_a), (amp * phase, occ_b)]
        self.branches = new_branches

    def build_initial_state(self):
        self.check_state()
        psi = sum(amp * tensor_prod([basis_state(b, 2) for b in occ])
                  for amp, occ in self.branches)
        return psi.unit() if hasattr(psi, 'unit') else psi


    def build_H(self):

        if self.h_type == "spin":
            return self.build_spin_h()
        elif self.h_type == "exact_dw":
            return self.build_exact_dw_h()
        elif self.h_type == "approx_dw":
            return self.build_approx_dw_h()

    def prod(self,iterable):
        return reduce(operator.mul, iterable, 1)

    def string(self,k):
        return self.prod(Z(i)*Z(i+self.N) for i in range(k))
    

    def build_spin_h(self):
        N = self.N
        #Hopping
        H_t_up = -self.t/2*sum(X(i)*X(i+1)+Y(i)*Y(i+1) for i in range(N-1))
        H_t_down = -self.t/2*sum(X(i)*X(i+1)+Y(i)*Y(i+1) for i in range(N, self.nqubits-1))

        #F-H interaction
        H_U = self.U/4*sum((1-Z(i))*(1-Z(i+N)) for i in range(N))

        #Chemical potential
        U_mu = -self.mu/2*sum(1-Z(i) for i in range(self.nqubits))

        #Electric field
        H_h = -self.h*self.b*sum(self.string(i) for i in range(N))

        #Gauge field
        H_eta = self.eta*self.b/4*sum(self.string(i)*(1-Z(i))*(1-Z(i+N)) for i in range(N))

        return H_t_up + H_t_down + H_U + U_mu + H_h + H_eta

    def build_exact_dw_h(self):
        N = self.N
        #Hopping
        H_t_up = -self.t/2*sum(X(i)*(1-Z(i-1)*Z(i+1)) for i in range(1,N-1))
        H_t_down = -self.t/2*sum(X(i)*(1-Z(i-1)*Z(i+1)) for i in range(N+1, self.nqubits-1))

        #F-H interaction
        H_U = self.U/4*sum((1-Z(i)*Z(i+1))*(1-Z(i+N)*Z(i+N+1)) for i in range(N-1))

        #Chemical potential
        U_mu_up = -self.mu/2*sum((1-Z(i)*Z(i+1)) for i in range(N-1))
        U_mu_down = -self.mu/2*sum((1-Z(i)*Z(i+1)) for i in range(N, self.nqubits-1))

        #Electric field
        H_h = -self.h*sum(Z(i)*Z(i+N) for i in range(N-1))

        #Gauge field
        H_eta = self.eta/4*sum((Z(i)-Z(i+1))*(Z(i+N)-Z(i+N+1)) for i in range(N-1))

        return H_t_up + H_t_down + H_U + U_mu_up + U_mu_down + H_h + H_eta

    def build_approx_dw_h(self):
        N = self.N
        #Hopping
        H_t_up   = -self.t*sum(X(i) for i in range(1,N-1))
        H_t_down = -self.t*sum(X(i) for i in range(N+1, self.nqubits-1))

        #F-H interaction
        H_U = self.U/4*sum((1-Z(i)*Z(i+1))*(1-Z(i+N)*Z(i+N+1)) for i in range(N-1))

        #Chemical potential
        U_mu_up = -self.mu/2*sum((1-Z(i)*Z(i+1)) for i in range(N-1))
        U_mu_down = -self.mu/2*sum((1-Z(i)*Z(i+1)) for i in range(N, self.nqubits-1))

        #Electric field
        H_h = -self.h*sum(Z(i)*Z(i+N) for i in range(N-1))

        #Gauge field
        H_eta = self.eta/4*sum((Z(i)-Z(i+1))*(Z(i+N)-Z(i+N+1)) for i in range(N-1))

        #DW coupling
        H_J_up   = self.J*sum(Z(i)*Z(i+1) for i in range(N-1))
        H_J_down = self.J*sum(Z(i)*Z(i+1) for i in range(N, self.nqubits-1))

        return H_t_up + H_t_down + H_U + U_mu_up + U_mu_down + H_h + H_eta + H_J_up + H_J_down



    def check_initialization(self):
        """Validate the parameters. Hard errors raise ValueError; doubtful choices warn."""
        if self.h_type not in self.VALID_TYPES:
            raise ValueError(f"h_type must be one of {self.VALID_TYPES}, got {self.h_type!r}")
        if isinstance(self.length, bool) or not isinstance(self.length, (int, np.integer)) or self.length < 2:
            raise ValueError(f"L must be an integer >= 2, got {self.length!r}")
        if self.b not in (1, -1):
            raise ValueError(f"b must be +1 or -1, got {self.b!r}")
        for name in ("t", "U", "mu", "eta", "h"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) \
                    or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real number, got {value!r}")

        if self.h_type == "approx_dw":
            if self.J is None or not np.isfinite(self.J):
                raise ValueError("approx_dw needs a finite J")
            if abs(self.J) < 10 * abs(self.t):
                warnings.warn(f"|J| = {abs(self.J)} < 10|t|: corrections of order t^2/|J| may be large")
            if self.U != 0:
                warnings.warn("U is a 4-body term in the DW encoding: fine to simulate, not native on hardware")
        elif self.J not in (None, 0):
            raise ValueError(f"J only exists in approx_dw, got J = {self.J} for {self.h_type}")

        if self.nqubits > 24:
            gb = 2**self.nqubits * 30 * 16 / 1e9
            warnings.warn(f"{self.nqubits} qubits: a full-space sparse matrix needs roughly {gb:.0f} GB")

    def check_state(self):
        """Called by build_initial_state: branches must be valid bitstrings in the model's b sector."""
        if not self.branches or sum(abs(amp)**2 for amp, _ in self.branches) == 0:
            raise ValueError("the initial state has no branch with non-zero amplitude")
        for _, occ in self.branches:
            if len(occ) != self.nqubits or set(occ) - {0, 1}:
                raise ValueError(f"branch {occ} must have {self.nqubits} entries in {{0, 1}}")
            if self.h_type != "spin":
                N = self.N
                b_state = (1 - 2*occ[0]) * (1 - 2*occ[N])      # Z_0,up * Z_0,dn
                if b_state != self.b:
                    raise ValueError(f"branch {occ} has b = {b_state:+d}, the model has b = {self.b:+d}")


    
                        



