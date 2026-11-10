"""Reproduce Figs. 2 and 3 of the IBM Fermi-Hubbard paper (arXiv:2605.04025)
using the StandardFermiHubbard (standard / direct site) encoding.

Fig. 2: Néel initial state with a central vacancy, evolved at various U/t.
        Plots charge tracer C^c_i(t) and spin tracer C^s_i(t) and extracts
        wavefront velocities.
Fig. 3: Néel initial state with and without a central vacancy. Plots per-site
        spin-up occupation heatmap and per-site time slices of <n_up>, <n_down>.

The standard encoding uses nqubits = 2 * chain_length so this will only run
for small L (qutip is exact). Set L, U_values, T, nsteps from the CLI block
at the bottom.

Conventions of StandardFermiHubbard:
  - |1> = occupied, so <n> = (1 - <Z>) / 2
  - 'up' chain: sites 0..L-1   -> spin-up
  - 'down' chain: sites L..2L-1 -> spin-down
  - Hopping coefficient = self.t (the kinetic amplitude, paper's t_h)
  - Interaction term is U * Z_up_i * Z_down_i; if you want the full
    U n_up n_down = (U/4)(1-Z_up)(1-Z_down) you should add a chemical
    potential e = -U/4 to absorb the linear-Z parts. Either way, ZZ alone
    captures the qualitative interaction physics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import theilslopes

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.qutip_model_encodings import StandardFermiHubbard


# ----------------------------------------------------------------------------
# Initial-state helpers
# ----------------------------------------------------------------------------

def neel_state(model, vacancy_site=None):
    """Place a Néel pattern |↑↓↑↓...⟩ on the chains.

    Site i (0-indexed) holds spin-up if i is even, spin-down if i is odd.
    If vacancy_site is given, the spin at that site is removed instead.
    """
    L = model.chain_length
    model.initialize_chain()
    for i in range(L):
        if i == vacancy_site:
            continue
        if i % 2 == 0:
            model.place_single_excitation("up", i)
        else:
            model.place_single_excitation("down", i)


# ----------------------------------------------------------------------------
# Observable extraction (from <Z> measurements)
# ----------------------------------------------------------------------------

def occupations(measurements_Z, chain_length):
    """Return n_up[t, i], n_down[t, i] from <Z> measurements.

    measurements_Z has shape (n_time, n_observables_in_string).
    For pauli string 'Z', build_pauli_observables iterates first over the
    upper chain (sites 0..L-1) then the lower chain (sites L..2L-1).
    """
    Z = np.asarray(measurements_Z)
    n = (1.0 - Z) / 2.0
    n_up = n[:, :chain_length]
    n_down = n[:, chain_length:]
    return n_up, n_down


def sz_per_site(n_up, n_down):
    """S^z_i = (n_up - n_down) / 2 per site."""
    return 0.5 * (n_up - n_down)


def charge_tracer(n_up, n_down):
    """C^c_i(t) = <n_up + n_down>(t) - <n_up + n_down>(0)."""
    rho = n_up + n_down
    return rho - rho[0]


def spin_tracer(model, sim_states, vacancy_site):
    """C^s_i(t) = 4 ( <S^z_i S^z_{i*}> - <S^z_i><S^z_{i*}> ).

    Built directly from qutip states because the connected piece requires
    a two-site correlator, not just single-site expectations.
    """
    import qutip as qt

    L = model.chain_length
    nq = model.nqubits
    states = sim_states.states
    n_time = len(states)

    # S^z_i = (n_up_i - n_down_i)/2 = ((I - Z_up_i) - (I - Z_down_i)) / 4
    #       = (Z_down_i - Z_up_i) / 4
    sz_ops = []
    for i in range(L):
        up_idx = i
        down_idx = i + L
        sz_ops.append(0.25 * (model.sz_list[down_idx] - model.sz_list[up_idx]))

    sz_star = sz_ops[vacancy_site]

    C = np.zeros((n_time, L))
    for t_idx, state in enumerate(states):
        sz_star_exp = qt.expect(sz_star, state)
        for i in range(L):
            corr = qt.expect(sz_ops[i] * sz_star, state)
            sz_i_exp = qt.expect(sz_ops[i], state)
            C[t_idx, i] = 4.0 * (corr - sz_i_exp * sz_star_exp)
    return C


# ----------------------------------------------------------------------------
# Wavefront detection (paper supplement, Sec. V A)
# ----------------------------------------------------------------------------

def detect_wavefront_velocity(C, t_list, vacancy_site, smoothing_sigma=1.0,
                              threshold_frac=0.30, force_origin=False):
    """Detect the wavefront in a tracer correlator C[t, i].

    Steps per time slice:
      1. Spatially Gaussian-smooth |C(t, ·)|.
      2. Walk outward from vacancy_site on both sides past the peak.
      3. Find the innermost site where the signal drops below
         threshold_frac * peak.
      4. Average the left/right distances to give x_wf(t).
    Then Theil-Sen regression on x_wf(t) vs t.
    """
    n_time, L = C.shape
    positions = np.full(n_time, np.nan)

    for k in range(n_time):
        signal = gaussian_filter1d(np.abs(C[k]), sigma=smoothing_sigma)
        peak = signal.max()
        if peak <= 1e-12:
            continue
        thresh = threshold_frac * peak

        # walk right
        right = None
        for i in range(vacancy_site, L - 1):
            if signal[i] >= thresh and signal[i + 1] < thresh:
                # linear interp
                frac = (signal[i] - thresh) / (signal[i] - signal[i + 1] + 1e-15)
                right = (i + frac) - vacancy_site
                break
        # walk left
        left = None
        for i in range(vacancy_site, 0, -1):
            if signal[i] >= thresh and signal[i - 1] < thresh:
                frac = (signal[i] - thresh) / (signal[i] - signal[i - 1] + 1e-15)
                left = vacancy_site - (i - frac)
                break

        dists = [d for d in (left, right) if d is not None]
        if dists:
            positions[k] = np.mean(dists)

    mask = ~np.isnan(positions)
    if mask.sum() < 2:
        return float("nan"), positions

    t_fit = t_list[mask]
    x_fit = positions[mask]
    if force_origin:
        # least-squares slope through origin (robust-ish): use Theil-Sen on
        # x/t to estimate slope.
        slopes = x_fit[t_fit > 0] / t_fit[t_fit > 0]
        v = float(np.median(slopes)) if slopes.size else float("nan")
    else:
        slope, _, _, _ = theilslopes(x_fit, t_fit)
        v = float(slope)
    return v, positions


# ----------------------------------------------------------------------------
# Figure 2: spin-charge separation
# ----------------------------------------------------------------------------

def run_fig2(chain_length=7, U_values=(0.0, 2.0, 8.0, 14.0),
             t_hop=1.0, e=0.0, T=4.0, nsteps=120, outdir="."):
    """Reproduce Fig. 2: charge & spin tracer heatmaps for a Néel state with
    a central vacancy across several U/t values, plus extracted velocities.
    """
    L = chain_length
    vacancy_site = L // 2
    t_list = np.linspace(0.0, T, nsteps)

    results = []
    for U in U_values:
        model = StandardFermiHubbard(L, t_hop, U, e, T, nsteps)
        neel_state(model, vacancy_site=vacancy_site)
        sim, meas = model.run(pauli_strings=["Z"])
        n_up, n_down = occupations(meas["Z"], L)
        Cc = charge_tracer(n_up, n_down)
        Cs = spin_tracer(model, sim, vacancy_site)
        vc, _ = detect_wavefront_velocity(Cc, t_list, vacancy_site,
                                          force_origin=True)
        vs, _ = detect_wavefront_velocity(Cs, t_list, vacancy_site,
                                          force_origin=True)
        results.append((U, Cc, Cs, vc, vs))

    # heatmaps
    n_cols = len(U_values)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.5),
                             sharex=True, sharey=True)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    def _overlay_cone(ax, v, color="yellow"):
        """Draw the linear light cone x = vacancy_site ± v·t over the data."""
        t_max = T
        # clip so the line doesn't run off the lattice
        if v > 0:
            t_max = min(T, (L - 1 - vacancy_site) / v, vacancy_site / v)
        ax.plot([vacancy_site, vacancy_site + v * t_max],
                [0, t_max], "--", color=color, lw=1.6)
        ax.plot([vacancy_site, vacancy_site - v * t_max],
                [0, t_max], "--", color=color, lw=1.6)
        ax.axvline(vacancy_site, color=color, lw=0.5, alpha=0.4)

    for col, (U, Cc, Cs, vc, vs) in enumerate(results):
        # charge: signed signal, diverging cmap centered on 0
        vmax_c = np.max(np.abs(Cc)) or 1.0
        axes[0, col].imshow(Cc, aspect="auto", origin="lower",
                            extent=[0, L - 1, 0, T],
                            cmap="RdBu_r", vmin=-vmax_c, vmax=vmax_c)
        axes[0, col].set_title(f"U/t = {U:g}\n$v_c$ = {vc:.2f}")
        _overlay_cone(axes[0, col], vc, color="black")

        # spin: also signed; use a perceptually uniform sequential map on |C|
        # to make the wavefront edge pop out against the background.
        absCs = np.abs(Cs)
        vmax_s = absCs.max() or 1.0
        axes[1, col].imshow(absCs, aspect="auto", origin="lower",
                            extent=[0, L - 1, 0, T],
                            cmap="magma", vmin=0, vmax=vmax_s)
        axes[1, col].set_title(f"$v_s$ = {vs:.2f}")
        _overlay_cone(axes[1, col], vs, color="cyan")
        axes[1, col].set_xlabel("Site")

    axes[0, 0].set_ylabel(r"Charge $C^c_i(t)$    Time $t$")
    axes[1, 0].set_ylabel(r"Spin $C^s_i(t)$    Time $t$")
    fig.suptitle(f"Fig. 2 reproduction (L={L}, standard encoding)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_tracers.png"), dpi=150)

    # velocity vs U
    Us = np.array([r[0] for r in results])
    vcs = np.array([r[3] for r in results])
    vss = np.array([r[4] for r in results])
    fig2, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    ax[0].plot(Us, vcs, "o-")
    ax[0].set_xlabel("U / t"); ax[0].set_ylabel(r"$v_c$")
    ax[1].plot(Us, vss, "s-", color="tab:purple")
    ax[1].set_xlabel("U / t"); ax[1].set_ylabel(r"$v_s$")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(vss != 0, vcs / vss, np.nan)
    ax[2].plot(Us, ratio, "d-", color="tab:green")
    ax[2].set_xlabel("U / t"); ax[2].set_ylabel(r"$v_c / v_s$")
    fig2.tight_layout()
    fig2.savefig(os.path.join(outdir, "fig2_velocities.png"), dpi=150)

    return results


# ----------------------------------------------------------------------------
# Figure 3: Néel state quench, with and without central vacancy
# ----------------------------------------------------------------------------

def run_fig3(chain_length=7, U=-2.0, t_hop=1.0, e=0.0, T=6.0, nsteps=180,
             slice_sites=None, outdir="."):
    """Reproduce Fig. 3: per-site spin-up occupation heatmap and per-site
    time slices, for the Néel state with and without a central vacancy.
    """
    L = chain_length
    t_list = np.linspace(0.0, T, nsteps)
    vacancy_site = L // 2
    if slice_sites is None:
        # spread three sites across the chain like the paper's 20/29/40 on L=60
        slice_sites = [L // 3, vacancy_site, (2 * L) // 3]

    cases = [("Néel", None), ("Néel + vacancy", vacancy_site)]

    fig, axes = plt.subplots(4, 2, figsize=(11, 11),
                             gridspec_kw={"height_ratios": [3, 1, 1, 1]})

    for col, (label, vac) in enumerate(cases):
        model = StandardFermiHubbard(L, t_hop, U, e, T, nsteps)
        neel_state(model, vacancy_site=vac)
        _, meas = model.run(pauli_strings=["Z"])
        n_up, n_down = occupations(meas["Z"], L)

        im = axes[0, col].imshow(n_up.T, aspect="auto", origin="lower",
                                 extent=[0, T, 0, L - 1],
                                 cmap="RdBu_r", vmin=0, vmax=1)
        axes[0, col].set_title(f"{label} — ⟨n↑⟩(i, t),  U/t={U:g}")
        axes[0, col].set_ylabel("Site")
        if vac is not None:
            # rough causal cone t_h = 1
            axes[0, col].plot([0, T], [vac, vac - 2 * T], "k-", lw=0.8)
            axes[0, col].plot([0, T], [vac, vac + 2 * T], "k-", lw=0.8)
        plt.colorbar(im, ax=axes[0, col], label="⟨n↑⟩")

        for row, site in enumerate(slice_sites, start=1):
            axes[row, col].plot(t_list, n_up[:, site], "^-",
                                color="tab:blue", label=r"$n_\uparrow$",
                                markersize=3)
            axes[row, col].plot(t_list, n_down[:, site], "v-",
                                color="tab:red", label=r"$n_\downarrow$",
                                markersize=3)
            axes[row, col].set_ylim(-0.05, 1.05)
            axes[row, col].set_ylabel(f"site {site}")
            axes[row, col].grid(alpha=0.3)
            if row == 1 and col == 1:
                axes[row, col].legend(loc="upper right", fontsize=8)
        axes[-1, col].set_xlabel(r"Time $t$")

    fig.suptitle(f"Fig. 3 reproduction (L={L}, standard encoding)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_neel_quench.png"), dpi=150)
    return fig


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    outdir = os.path.join(os.path.dirname(__file__), "ibm_paper_outputs")
    os.makedirs(outdir, exist_ok=True)

    # Fig. 2: keep L small (qutip is exact, 2L qubits)
    run_fig2(chain_length=7, U_values=(0.0, 2.0, 8.0, 14.0),
             t_hop=1.0, e=0.0, T=3.5, nsteps=120, outdir=outdir)

    # Fig. 3: smaller U so the staggered pattern actually melts in time T
    run_fig3(chain_length=7, U=-2.0, t_hop=1.0, e=0.0,
             T=6.0, nsteps=180, outdir=outdir)

    plt.show()
