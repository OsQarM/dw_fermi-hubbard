"""Initial Fock states and time evolution with QiliSim."""
import numpy as np
from qilisdk.analog import Schedule
from qilisdk.backends import QiliSim
from qilisdk.core import basis_state, tensor_prod
from qilisdk.functionals import AnalogEvolution
from qilisdk.readout import Readout

from hamiltonian import UP, DN, positions


def fock_bits(model, up, down, count, b=1):
    """Qubit bitstring (0 -> Z=+1, 1 -> Z=-1) for site occupations up[i], down[i] in {0, 1}.

    site counting: one bit per qubit, 1 = occupied.
    wall counting: left edge qubit fixed (up: Z=+1, down: Z=b); each occupied site i
                   puts a domain wall on bond i, i.e. flips every qubit to its right.
    """
    L = positions(model.N, count)
    bits = []
    for s, occ in ((UP, up), (DN, down)):
        if len(occ) != L:
            raise ValueError(f"occupation list must have {L} entries, got {occ}")
        if count == "site":
            bits += list(occ)
        else:
            bit = 0 if (s == UP or b == 1) else 1
            chain = [bit]
            for n in occ:
                bit ^= n
                chain.append(bit)
            bits += chain
    return bits


def initial_state(model, branches, count, b=1):
    """Normalized superposition sum_k amplitude_k |up_k, down_k> (branches as in config.yaml)."""
    psi = sum(br["amplitude"] * tensor_prod([basis_state(x, 2)
                                             for x in fock_bits(model, br["up"], br["down"], count, b)])
              for br in branches)
    return psi.unit()


def evolve(model, psi0, observables, T, dt):
    """Evolve psi0 under the constant model.hamiltonian() with QiliSim.

    Returns times (n_t,) and expectation values (n_t, len(observables)).
    """
    H = model.hamiltonian()
    if H.nqubits != model.nqubits:
        raise ValueError(f"H acts on {H.nqubits} qubits, the model has {model.nqubits}: "
                         "switch on a term that touches every qubit (e.g. a hopping or n term)")
    schedule = Schedule.constant(H, total_time=T, dt=dt)
    evolution = AnalogEvolution(schedule=schedule, initial_state=psi0, store_intermediate_results=True)
    result = QiliSim().execute(evolution, Readout().with_expectation(observables=observables))
    values = np.real(np.array(result.get_intermediate_expectation_values()))
    times = np.linspace(0, T, len(values))
    return times, values
