"""Static properties: sector selection, lowest eigenpairs, expectation values.

Workflow: model.hamiltonian() -> matrix() -> sector() -> lowest_states() -> expect().
count = "site" or "wall" says what a particle is (see hamiltonian.occupation).
All vectors live in the full 2**nqubits space, in qilisdk's ordering (qubit 0 = leftmost factor).
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from hamiltonian import UP, DN, Z, occupation, positions, number_op, string, qubit


def matrix(op, nqubits):
    """Sparse matrix of a qilisdk operator on nqubits qubits.

    qilisdk sizes the matrix by the highest qubit an operator touches,
    so operators on the first qubits only are padded with identities on the right.
    """
    M = sp.csr_matrix(op.to_matrix())
    missing = nqubits - op.nqubits
    return sp.kron(M, sp.identity(2**missing), format="csr") if missing > 0 else M


def diagonal(op, nqubits):
    return matrix(op, nqubits).diagonal().real


def _z(model, k, s):
    return diagonal(Z(qubit(model.N, k, s)), model.nqubits)


def boundary_mask(model, count, b=1):
    """wall counting: left edge qubits fixed, Z_0,up = +1 and Z_0,dn = b. site counting: no constraint.

    Flipping both chains is a symmetry of the wall terms, so fixing Z_0,up = +1
    only removes an exact copy of the spectrum.
    """
    mask = np.ones(2**model.nqubits, dtype=bool)
    if count == "wall":
        mask &= np.isclose(_z(model, 0, UP), 1) & np.isclose(_z(model, 0, DN), b)
    return mask


def sector(model, N_up, N_dn, count, b=1):
    """Indices of the basis states with N_up, N_dn particles (and the boundary sector, for walls)."""
    mask = boundary_mask(model, count, b)
    mask &= np.isclose(diagonal(number_op(model.N, UP, count), model.nqubits), N_up)
    mask &= np.isclose(diagonal(number_op(model.N, DN, count), model.nqubits), N_dn)
    return np.flatnonzero(mask)


def edge_sector(model, N_up, N_dn, b=1):
    """Wall counting with both edge qubits fixed: N_s is fixed only modulo 2.

    This is the symmetry sector of the approximate DW model (x_field, zz_chain),
    whose transverse field creates and removes pairs of walls.
    """
    mask = boundary_mask(model, "wall", b)
    for s, N, z0 in ((UP, N_up, 1), (DN, N_dn, b)):
        mask &= np.isclose(_z(model, model.N - 1, s), z0 * (-1) ** N)
    return np.flatnonzero(mask)


def lowest_states(H, idx, k=1, seed=0):
    """k lowest eigenpairs of the sparse matrix H restricted to the basis states idx.

    Dense diagonalization for small sectors, Lanczos (eigsh) otherwise.
    Eigenvectors are returned embedded in the full space, as columns.
    """
    Hs = H[idx][:, idx]
    if Hs.shape[0] <= 500:
        E, V = np.linalg.eigh(Hs.toarray())
        E, V = E[:k], V[:, :k]
    else:
        v0 = np.random.default_rng(seed).random(Hs.shape[0])
        E, V = eigsh(Hs, k=k, which="SA", v0=v0)
        order = np.argsort(E)
        E, V = E[order], V[:, order]
    full = np.zeros((H.shape[0], V.shape[1]), dtype=complex)
    full[idx] = V
    return E, full


def spectrum(H, idx):
    """All eigenvalues of H in the sector idx (small sectors only)."""
    return np.linalg.eigvalsh(H[idx][:, idx].toarray())


def expect(op, psi, nqubits):
    return float(np.real(np.vdot(psi, matrix(op, nqubits) @ psi)))


def site_profile(model, psi, count, b=1):
    """<n_up>, <n_dn>, <n_up n_dn> and <eta> on every fermion site.

    eta: b * S_k for site counting, Z_k,up Z_k,dn for wall counting.
    """
    N, nq = model.N, model.nqubits
    n = lambda k, s: occupation(N, k, s, count)
    eta = lambda k: b * string(N, k) if count == "site" else Z(qubit(N, k, UP)) * Z(qubit(N, k, DN))
    sites = range(positions(N, count))
    return {"n_up": np.array([expect(n(k, UP), psi, nq) for k in sites]),
            "n_dn": np.array([expect(n(k, DN), psi, nq) for k in sites]),
            "double": np.array([expect(n(k, UP) * n(k, DN), psi, nq) for k in sites]),
            "eta": np.array([expect(eta(k), psi, nq) for k in sites])}
