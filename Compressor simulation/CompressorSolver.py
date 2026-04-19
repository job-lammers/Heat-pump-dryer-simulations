"""
CompressorSolver.py
===================
Driver script for the multi-stage centrifugal compressor model.

Workflow
--------
1. Define the working fluid mixture and operating boundary conditions.
2. Sweep over a range of specific speeds (N_s) to identify the design
   point that maximises the overall isentropic efficiency.
3. Re-run the optimal case with detailed per-stage output.

The compressor sizing follows a two-pass approach:
- **First pass**: iterate on stage efficiency until convergence.
- **Second pass**: recompute stage enthalpies with the converged per-stage
  efficiencies to obtain consistent power and overall performance.

Dependencies
------------
- numpy
- CoolProp (with REFPROP backend)
- Compressor  (see Compressor.py)

Usage
-----
Run directly::

    python CompressorSolver.py

All user-adjustable parameters are collected in the "Configuration" section
below.

Author
------
J.H. Lam
"""

from Compressor import Compressor
from CoolProp import AbstractState as AS
from CoolProp import HmassP_INPUTS, PSmass_INPUTS, PT_INPUTS
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

# ---- Working fluid (REFPROP mixture) ----
# Uncomment the mixture pair relevant to your heat-pump cascade cycle.

# Cascade pair 2 – upper cycle
# mixture       = ["IsoPentane", "Dimethyl Ether"]
# mole_fraction = [0.55, 0.45]

# Cascade pair 2 – lower cycle  (active)
mixture       = ["Dimethyl Ether", "IsoPentane"]
mole_fraction = [0.62, 0.38]

# ---- Operating boundary conditions ----
# Upper cycle example (commented out):
# m_dot   = 1.591          # mass flow rate            [kg/s]
# P_start = 10.59e5        # compressor inlet pressure [Pa]
# T_start = 127.4          # compressor inlet temp.    [°C]
# P_end   = 53.51e5        # compressor outlet pressure[Pa]

m_dot         = 1.118      # mass flow rate             [kg/s]
P_start       = 7.13e5     # compressor inlet pressure  [Pa]
T_start       = 80.0       # compressor inlet temp.     [°C]
P_end         = 18.16e5    # compressor outlet pressure [Pa]

# Initial guess for stage isentropic efficiency (updated iteratively)
eta_is_start  = 0.8

# ---- Compressor design parameters ----
n_stages        = 2         # number of stages                     [-]
Ratio_inlet     = 0.3       # hub-to-tip diameter ratio D_1h/D_1t  [-]
Ratio_inoutlet  = 0.5       # tip-to-exit diameter ratio D_1t/D_2  [-]
Ratio_diffuser  = 1.6       # diffuser-to-exit diameter D_3/D_2    [-]
b_star          = 0.95      # diffuser inlet blockage factor        [-]
epsilon         = 0.25      # jet-to-wake momentum ratio            [-]
Z               = 18        # number of impeller blades             [-]
x               = 0.002     # blade thickness                       [m]
t               = 0.0005    # tip clearance                         [m]
RatioC_u2       = 0.65      # velocity ratio C_u2 / U_2             [-]
RatioC_r2       = 0.30      # velocity ratio C_r2 / U_2             [-]

# ---- Specific speed sweep settings ----
Ns_min  = 0.6               # lower bound of N_s sweep              [-]
Ns_max  = 1.3               # upper bound of N_s sweep              [-]
Ns_step = 0.01              # step size                             [-]

# =============================================================================
# Fluid initialisation
# =============================================================================

fluid = AS("REFPROP", "&".join(mixture))
fluid.set_mole_fractions(mole_fraction)


# =============================================================================
# Core simulation function
# =============================================================================

def run_compressor_train(N_s: float, verbose: bool = True):
    """
    Simulate the multi-stage compressor for a given specific speed.

    The function performs two passes over the stage sequence:

    1. **First pass** – converge per-stage isentropic efficiency using the
       loss model inside :class:`Compressor`.
    2. **Second pass** – recompute stage enthalpies with the converged
       efficiencies to obtain consistent power consumption and overall
       isentropic efficiency.

    Parameters
    ----------
    N_s : float
        Dimensionless specific speed used to size the first-stage rotational
        speed [-].
    verbose : bool, optional
        If True, print per-stage efficiency and full compressor state.
        Default is True.

    Returns
    -------
    eta_overall : float
        Overall isentropic efficiency of the compressor train [-].
        Returns 0 if the simulation fails.
    etac_list : list of float
        Converged isentropic efficiency for each stage [-].
    comps : list of Compressor
        Compressor stage objects (one per stage).
    final_state : dict
        Dictionary with keys ``'P_out'`` [Pa], ``'T_out'`` [K],
        ``'h_out'`` [J/kg] for the final stage outlet.
    power_per_stage : numpy.ndarray
        Shaft power consumption per stage [kW].

    Notes
    -----
    Returns ``(0, 0, 0, 0, 0)`` if a numerical exception occurs (e.g. the
    specific speed produces an infeasible geometry).
    """
    PR_stage = (P_end / P_start) ** (1.0 / n_stages)  # per-stage pressure ratio

    P_in       = P_start
    T_in       = T_start + 273.15        # convert inlet temperature to K
    first_stage = True
    omega       = 0.0
    max_iter    = 50                     # max efficiency iterations per stage
    tol         = 1e-2                   # convergence tolerance on eta_is

    etac_list = []
    comps     = []

    try:
        # ==============================================================
        # First pass: converge per-stage efficiency
        # ==============================================================
        for i in range(n_stages):
            eta_is = eta_is_start

            for _ in range(max_iter):
                P_out = P_in * PR_stage

                comp = Compressor(
                    fluid,
                    m_dot,
                    P_in, T_in,
                    P_out,
                    eta_is,
                    N_s,
                    first_stage,
                    omega,
                    RatioC_u2,
                    RatioC_r2,
                    Z, x, t,
                    Ratio_inoutlet,
                    Ratio_inlet,
                    b_star,
                    epsilon,
                    Ratio_diffuser,
                )

                eta_calc = comp.eta_calculated

                if abs(eta_calc - eta_is) / eta_is < tol:
                    break

                eta_is = eta_calc  # update guess and iterate

            etac_list.append(eta_calc)
            comps.append(comp)

            if verbose:
                print(f"\nStage {i + 1}  –  isentropic efficiency = {eta_calc:.3f}")
                print(comp)

            # Propagate shaft speed and thermodynamic state to next stage
            first_stage = False
            omega       = comp.omega
            P_in        = P_out
            fluid.update(HmassP_INPUTS, comp.h_out, P_in)
            T_in = fluid.T()

        # ==============================================================
        # Second pass: compute power and overall efficiency
        # ==============================================================
        P_in  = P_start
        T_in  = T_start + 273.15
        power_per_stage = np.zeros(n_stages)

        for i in range(n_stages):
            P_out = P_in * PR_stage

            fluid.update(PT_INPUTS, P_in, T_in)
            h_in = fluid.hmass()
            s_in = fluid.smass()

            fluid.update(PSmass_INPUTS, P_out, s_in)
            h_out_is = fluid.hmass()

            # Real specific work using converged per-stage efficiency
            h_out = (h_out_is - h_in) / etac_list[i] + h_in

            power_per_stage[i] = (h_out - h_in) * m_dot / 1e3  # [kW]

            fluid.update(HmassP_INPUTS, h_out, P_out)
            T_in = fluid.T()
            P_in = P_out

        # ---- Overall isentropic efficiency ----
        # Use inlet-to-outlet isentropic path at the global pressure ratio
        fluid.update(PT_INPUTS, P_start, T_start + 273.15)
        h_start = fluid.hmass()
        s_start = fluid.smass()

        fluid.update(PSmass_INPUTS, P_end, s_start)
        h_end_is = fluid.hmass()

        fluid.update(PT_INPUTS, P_end, T_in)
        h_end = fluid.hmass()

        eta_overall = (h_end_is - h_start) / (h_end - h_start)

        final_state = {"P_out": P_end, "T_out": T_in, "h_out": h_out}
        return eta_overall, etac_list, comps, final_state, power_per_stage

    except Exception:
        # Return sentinel values so the sweep can continue gracefully
        return 0, 0, 0, 0, 0


# =============================================================================
# Specific-speed sweep
# =============================================================================

best = {
    "N_s"         : None,
    "eta_overall" : -1.0,
    "etac_list"   : None,
    "comps"       : None,
    "final_state" : None,
    "power_per_stage": None,
}

print("Sweeping specific speed N_s ...")
Ns = Ns_min
while Ns <= Ns_max + 1e-12:   # small tolerance to include Ns_max
    eta_overall, etac_list, comps, final_state, power_per_stage = (
        run_compressor_train(Ns, verbose=False)
    )

    if eta_overall > best["eta_overall"]:
        best.update({
            "N_s"            : Ns,
            "eta_overall"    : eta_overall,
            "etac_list"      : etac_list,
            "comps"          : comps,
            "final_state"    : final_state,
            "power_per_stage": power_per_stage,
        })

    Ns = round(Ns + Ns_step, 5)   # round to avoid floating-point drift


# =============================================================================
# Results
# =============================================================================

print(
    f"\nOptimal N_s = {best['N_s']:.2f}  "
    f"|  Overall isentropic efficiency = {best['eta_overall']:.3f}\n"
)

print("=" * 60)
print("Detailed per-stage results for optimal N_s")
print("=" * 60)
run_compressor_train(best["N_s"], verbose=True)

print(
    f"\nOverall isentropic efficiency : {best['eta_overall']:.3f}"
)
print(
    f"Power consumption per stage   : "
    + "  |  ".join(f"Stage {i+1}: {p:.2f} kW"
                   for i, p in enumerate(best["power_per_stage"]))
)
