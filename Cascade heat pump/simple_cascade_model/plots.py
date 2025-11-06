# -*- coding: utf-8 -*-
"""
Plot utilities for the cascade heat pump.

- Ts diagram with saturation dome
- T–Q diagrams for all relevant heat exchangers
- Optional helper to plot two Ts curves from entropy/temperature data

@author: JHLam
"""
import matplotlib.pyplot as plt
import numpy as np
from .constants import KELVIN
from CoolProp import QT_INPUTS, PQ_INPUTS, PSmass_INPUTS, HmassP_INPUTS, PT_INPUTS


def plot_Ts_diagram_with_dome(cycle, Ref, stage, IHX, IHX_high, extended=True):
    """Plot Ts diagram of either the upper or lower cycle, including saturation dome."""
    N = 100
    ax = plt.gca()

    # === saturation dome ===
    T_min = 40 + 273.15
    T_max = Ref.T_critical()
    P_crit = Ref.p_critical()
    T_range = np.linspace(T_min, T_max, N)

    s_loop, T_loop = [], []
    for T in T_range:  # saturated liquid
        try:
            Ref.update(QT_INPUTS, 0, T)
            s_loop.append(Ref.smass())
            T_loop.append(T - KELVIN)
        except Exception:
            continue

    Ref.update(PT_INPUTS, P_crit, T_max)
    s_crit = Ref.smass()
    s_loop.append(s_crit)
    T_loop.append(T_max - KELVIN)

    for T in reversed(T_range):  # saturated vapor
        try:
            Ref.update(QT_INPUTS, 1, T)
            s_loop.append(Ref.smass())
            T_loop.append(T - KELVIN)
        except Exception:
            continue

    # === read cycle points from stage ===
    if stage == "Upper":
        P_cond = cycle.high["P3"]
        P_evap = cycle.high["P2"]
        Ref.update(PQ_INPUTS, P_evap, 1)
        s1, T1 = Ref.smass(), Ref.T()
        s2, T2 = cycle.high["s2"], cycle.high["T2"]
        s3, T3 = cycle.high["s3"], cycle.high["T3"]
        s4, h4, T4 = cycle.high["s4"], cycle.high["h4"], cycle.high["T4"]
        s5, h5, T5 = cycle.high["s5"], cycle.high["h5"], cycle.high["T5"]
        if IHX:
            s6, T6 = cycle.high["s6"], cycle.high["T6"]
    else:  # Lower
        P_cond = cycle.low["P8"]
        P_evap = cycle.low["P7"]
        Ref.update(PQ_INPUTS, P_evap, 1)
        s1, T1 = Ref.smass(), Ref.T()
        s2, T2 = cycle.low["s7"], cycle.low["T7"]
        s3, T3 = cycle.low["s8"], cycle.low["T8"]
        s4, h4, T4 = cycle.low["s9"], cycle.low["h9"], cycle.low["T9"]
        s5, h5, T5 = cycle.low["s10"], cycle.low["h10"], cycle.low["T10"]
        if IHX:
            s6, T6 = cycle.low["s11"], cycle.low["T11"]

    # === assemble process paths ===
    if not IHX:
        h_valve = h4
        s_evap1 = np.linspace(s5, s1, N)
        s_points = [s1, s2, s3, s4, s5]
        T_points = [T1, T2, T3, T4, T5]
        P_points = [P_evap, P_evap, P_cond, P_cond, P_evap]
    else:
        h_valve = h5
        s_evap1 = np.linspace(s6, s1, N)
        s_points = [s1, s2, s3, s4, s5, s6]
        T_points = [T1, T2, T3, T4, T5, T6]
        P_points = [P_evap, P_evap, P_cond, P_cond, P_cond, P_evap]

    # evaporator part (quality 1 → superheat)
    T_evap1 = np.array([Ref.update(PSmass_INPUTS, P_evap, s) or Ref.T() for s in s_evap1]) - KELVIN
    s_evap2 = np.linspace(s1, s2, N)
    T_evap2 = np.array([Ref.update(PSmass_INPUTS, P_evap, s) or Ref.T() for s in s_evap2]) - KELVIN

    # condenser part
    s_cond = np.linspace(s3, s4, N)
    T_cond = np.array([Ref.update(PSmass_INPUTS, P_cond, s) or Ref.T() for s in s_cond]) - KELVIN

    # expansion line
    p_vals = np.linspace(P_cond, P_evap, N)
    s_valve = np.empty(N)
    T_valve = np.empty(N)
    for i, p in enumerate(p_vals):
        try:
            Ref.update(HmassP_INPUTS, h_valve, p)
            s_valve[i] = Ref.smass()
            T_valve[i] = Ref.T() - KELVIN
        except Exception:
            continue

    if not IHX:
        s_all = np.concatenate([s_evap1, s_evap2, s_cond, s_valve])
        T_all = np.concatenate([T_evap1, T_evap2, T_cond, T_valve])
    else:
        s_IHX_high = np.linspace(s4, s5, N)
        T_IHX_high = np.array([Ref.update(PSmass_INPUTS, P_cond, s) or Ref.T() for s in s_IHX_high]) - KELVIN
        s_all = np.concatenate([s_evap1, s_evap2, s_cond, s_IHX_high, s_valve])
        T_all = np.concatenate([T_evap1, T_evap2, T_cond, T_IHX_high, T_valve])

    # === plotting ===
    plt.plot(s_all, T_all, label=f"{stage} cycle")
    plt.plot(s_loop, T_loop, "--", color="black", label="Saturation dome")

    # state numbering: upper starts at 1, lower gets shifted to not clash
    if stage == "Upper":
        start_idx = 1
    else:
        start_idx = 7 if IHX_high else 6

    if extended:
        for i, (s, T, P) in enumerate(zip(s_points, T_points, P_points), start=start_idx):
            ax.plot(s, T - KELVIN, "ro")
            txtT = f"$T_{{{i}}}$ = {T - KELVIN:.1f}"
            txtP = f"$P_{{{i}}}$ = {P/1e5:.2f}"
            fontsize = 20
            count = i - start_idx + 1

            if count == 1:
                ax.annotate(txtT, (s, T - KELVIN), xytext=(10, 0), textcoords="offset points",
                            fontsize=fontsize, color="red", ha="left", va="top")
                ax.annotate(txtP, (s, T - KELVIN), xytext=(10, -20), textcoords="offset points",
                            fontsize=fontsize, color="red", ha="left", va="top")
            elif count < 4:
                ax.annotate(txtT, (s, T - KELVIN), xytext=(10, 0), textcoords="offset points",
                            fontsize=fontsize, color="red", ha="left", va="bottom")
                ax.annotate(txtP, (s, T - KELVIN), xytext=(10, -20), textcoords="offset points",
                            fontsize=fontsize, color="red", ha="left", va="bottom")
            elif count == 4:
                ax.annotate(txtT, (s, T - KELVIN), xytext=(-10, 20), textcoords="offset points",
                            fontsize=fontsize, color="red", ha="right", va="bottom")
                ax.annotate(txtP, (s, T - KELVIN), xytext=(-10, 0), textcoords="offset points",
                            fontsize=fontsize, color="red", ha="right", va="bottom")
            elif count == 5 and IHX:
                ax.annotate(txtT, (s, T - KELVIN), xytext=(-10, 0), textcoords="offset points",
                            fontsize=fontsize, color="red", ha="right", va="bottom")
                ax.annotate(txtP, (s, T - KELVIN), xytext=(-10, -20), textcoords="offset points",
                            fontsize=fontsize, color="red", ha="right", va="bottom")
            else:
                ax.annotate(txtT, (s, T - KELVIN), xytext=(-10, -5), textcoords="offset points",
                            fontsize=fontsize, color="red", ha="left", va="top")
                ax.annotate(txtP, (s, T - KELVIN), xytext=(-10, -25), textcoords="offset points",
                            fontsize=fontsize, color="red", ha="left", va="top")
    else:
        for i, (s, T) in enumerate(zip(s_points, T_points), start=start_idx):
            plt.plot(s, T - KELVIN, "ro")
            plt.text(s - 5, T - KELVIN, f"{i}", fontsize=20, color="red", ha="right", va="bottom")

    # critical point
    plt.scatter(s_crit, T_max - KELVIN, label="Critical point", color="black", marker="*", s=70)

    plt.xlabel("Entropy [J/kgK]")
    plt.ylabel("Temperature [°C]")
    plt.grid(True)
    plt.legend()

    # ensure labels are inside limits
    ax.relim()
    for t in ax.texts:
        ax.update_datalim(
            t.get_window_extent(renderer=plt.gcf().canvas.get_renderer()).transformed(ax.transData.inverted())
        )
    ax.autoscale_view()
    plt.tight_layout()
    plt.show()


def plot_TQ_diagram_all(cycle, dryer, plt_sink, plt_shared, plt_source, IHX_high, IHX_low, HTF):
    """Plot T–Q diagrams for selected heat exchangers."""
    plt.figure(figsize=(10, 6))

    def plot_TQ_curve(h_array, T_array, m_dot, label, color):
        Q = (h_array - h_array[0]) * m_dot / 1e3   # kW (since h in J/kg)
        T_C = T_array - KELVIN
        plt.plot(Q, T_C, label=label, lw=2, color=color)

    # sink side (upper condenser + air/HTF)
    if plt_sink:
        plot_TQ_curve(np.flip(cycle.high["h_Cond"]), np.flip(cycle.high["T_Cond"]),
                      cycle.high["m_dot"], "Condenser high cycle", "black")
        if HTF:
            plot_TQ_curve(cycle.htf["h_sink"], cycle.htf["T_sink"],
                          cycle.htf["m_dot_sink"], "Heat transfer fluid", "purple")
        plot_TQ_curve(cycle.air["h_sink"], cycle.air["T_sink"],
                      dryer.m_da1, "Sink air stream", "red")

    # shared HEX between high condenser and low evaporator
    if plt_shared:
        plot_TQ_curve(np.flip(cycle.low["h_Cond"]), np.flip(cycle.low["T_Cond"]),
                      cycle.low["m_dot"], "Condenser low cycle", "blue")
        plot_TQ_curve(cycle.high["h_Evap"], cycle.high["T_Evap"],
                      cycle.high["m_dot"], "Evaporator high cycle", "darkred")

    # source side (air → low evap) + optional HTF
    if plt_source:
        plot_TQ_curve(np.flip(cycle.air["h_source"]), np.flip(cycle.air["T_source"]),
                      dryer.m_da4, "Source air stream", "darkblue")
        if HTF:
            plot_TQ_curve(cycle.htf["h_source"], cycle.htf["T_source"],
                          cycle.htf["m_dot_source"], "HTF evaporator", "orange")
        plot_TQ_curve(cycle.low["h_Evap"], cycle.low["T_Evap"],
                      cycle.low["m_dot"], "Evaporator low cycle", "green")

    # IHX high
    if IHX_high:
        plot_TQ_curve(np.flip(cycle.high["h_IHX_high"]), np.flip(cycle.high["T_IHX_high"]),
                      cycle.high["m_dot"], "IHX high hot", "red")
        plot_TQ_curve(cycle.high["h_IHX_low"], cycle.high["T_IHX_low"],
                      cycle.high["m_dot"], "IHX high cold", "blue")

    # IHX low
    if IHX_low:
        plot_TQ_curve(np.flip(cycle.low["h_IHX_high"]), np.flip(cycle.low["T_IHX_high"]),
                      cycle.low["m_dot"], "IHX low hot", "red")
        plot_TQ_curve(cycle.low["h_IHX_low"], cycle.low["T_IHX_low"],
                      cycle.low["m_dot"], "IHX low cold", "blue")

    plt.xlabel("Heat transfer [kW]")
    plt.ylabel("Temperature [°C]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


