"""Double-chain Fermi-Hubbard family as a single qilisdk spin Hamiltonian.

Every model is the same formula; a coupling set to 0 switches its term off.
The encoding only decides how n, eta and the hopping are written in Pauli operators.

    H = - t     sum_s sum_i  hop(i, s)                 exact, number-conserving hopping
        + U     sum_i  n_i,up n_i,dn                   Hubbard
        + U_eta sum_i  eta_i n_i,up n_i,dn             DW-native (gauge-dressed) Hubbard
        - h     sum_i  eta_i                           electric field (Ge & Nori)
        - mu    sum_i (n_i,up + n_i,dn)                chemical potential
        - Gamma sum_s sum_k X_k,s                      dw only: bare transverse field
        + J     sum_s sum_k Z_k,s Z_k+1,s              dw only: intra-chain Ising coupling

    Fermi-Hubbard: t, U        Ge & Nori: t, U, h        DW model: t, U_eta, h
    Approximate DW model: t = 0, Gamma = hopping amplitude, |J| >> Gamma.

Encodings (s = 0 for up, 1 for down; |0> has Z = +1; qubit 0 is the leftmost tensor factor)

    jw : L qubits per chain, qubit(i, s) = i + s*L.    |1> = occupied.
         n_i   = (1 - Z_i) / 2
         eta_i = b * prod_{j<i} Z_j,up Z_j,dn          (b is a free sign)
         hop   = (X_i X_i+1 + Y_i Y_i+1) / 2           (JW strings cancel: all up, then all down)

    dw : L+1 qubits per chain, qubit(k, s) = k + s*(L+1). Site i = bond between qubits i and i+1.
         n_i   = (1 - Z_i Z_i+1) / 2                   (a domain wall on bond i)
         eta_i = Z_i,up Z_i,dn                         (b = Z_0,up Z_0,dn is fixed by the state)
         hop   = X_k (1 - Z_k-1 Z_k+1) / 2, k = i+1    (moves a wall across qubit k; edge qubits never flip)
         Plain U is a 4-body ZZZZ term here: fine for simulation, not native on hardware.
"""
from dataclasses import dataclass

from qilisdk.analog import Hamiltonian, X, Y, Z, I 
from ruamel.yaml import YAML

UP, DN = 0, 1


@dataclass
class Params:
    L: int                 # number of fermion sites
    encoding: str = "jw"   # "jw" or "dw"
    t: float = 1.0
    U: float = 0.0
    U_eta: float = 0.0
    h: float = 0.0
    mu: float = 0.0
    b: int = 1             # jw: sign in eta. dw: boundary sector Z_0,dn (Z_0,up = +1 is fixed)
    Gamma: float = 0.0     # dw only
    J: float = 0.0         # dw only

    def __post_init__(self):
        if self.encoding not in ("jw", "dw"):
            raise ValueError(f"encoding must be 'jw' or 'dw', got {self.encoding!r}")
        if self.b not in (1, -1):
            raise ValueError("b must be +1 or -1")
        if self.encoding == "jw" and (self.Gamma or self.J):
            raise ValueError("Gamma and J exist only in the dw encoding")

    @property
    def per_chain(self):
        return self.L if self.encoding == "jw" else self.L + 1

    @property
    def nqubits(self):
        return 2 * self.per_chain


def load_config(path):
    with open(path) as f:
        return YAML(typ="safe").load(f)


# ---------------------------------------------------------------- building blocks

def qubit(p, k, s):
    return k + s * p.per_chain


def n_op(p, i, s):
    """Occupation of site i, spin s."""
    if p.encoding == "jw":
        return 0.5 * (I(0) - Z(qubit(p, i, s)))
    return 0.5 * (I(0) - Z(qubit(p, i, s)) * Z(qubit(p, i + 1, s)))


def eta_op(p, i):
    """eta_i = b (-1)^(number of particles on sites j < i)."""
    if p.encoding == "jw":
        eta = p.b * I(0)
        for j in range(i):
            eta = eta * Z(qubit(p, j, UP)) * Z(qubit(p, j, DN))
        return eta
    return Z(qubit(p, i, UP)) * Z(qubit(p, i, DN))


def hop_op(p, i, s):
    """c+_i c_i+1 + h.c. for spin s (sites i and i+1)."""
    if p.encoding == "jw":
        a, c = qubit(p, i, s), qubit(p, i + 1, s)
        return 0.5 * (X(a) * X(c) + Y(a) * Y(c))
    k = i + 1
    left, mid, right = qubit(p, k - 1, s), qubit(p, k, s), qubit(p, k + 1, s)
    return 0.5 * X(mid) * (I(0) - Z(left) * Z(right))


def number_op(p, s):
    return sum((n_op(p, i, s) for i in range(p.L)), Hamiltonian())


# ---------------------------------------------------------------- the Hamiltonian

def hamiltonian(p):
    H = Hamiltonian()

    for s in (UP, DN):
        for i in range(p.L - 1):
            H += -p.t * hop_op(p, i, s)

    for i in range(p.L):
        n_up, n_dn, eta = n_op(p, i, UP), n_op(p, i, DN), eta_op(p, i)
        H += p.U * n_up * n_dn
        H += p.U_eta * eta * n_up * n_dn
        H += -p.h * eta
        H += -p.mu * (n_up + n_dn)

    if p.encoding == "dw":
        for s in (UP, DN):
            for k in range(1, p.L):            # interior qubits only: b stays fixed
                H += -p.Gamma * X(qubit(p, k, s))
            for k in range(p.L):
                H += p.J * Z(qubit(p, k, s)) * Z(qubit(p, k + 1, s))

    return H
