# -*- coding: utf-8 -*-
"""
Plot utilities for the single-stage heat pump:
- Ts diagram with saturation dome
- T–Q diagrams for all streams
"""
import matplotlib.pyplot as plt
import numpy as np
from CoolProp import (
    QT_INPUTS,
    PQ_INPUTS,
    PSmass_INPUTS,
    HmassP_INPUTS,
    PT_INPUTS,
)
from .constants import KELVIN


def plot_Ts_diagram_with_dome(cycle, Ref, IHX, extended=True):
    """
    Plot the heat pump cycle on a Ts diagram, including the refrigerant
    saturation dome. If `extended=True`, annotate T_i and P_i.
    """
    N = 100
    ax = plt.gca()

    # === Saturation dome ===
    T_MinHigh = 40 + 273.15  # K
    T_MaxHigh = Ref.T_critical()
    P_crit = Ref.p_critical()
    T_range = np.linspace(T_MinHigh, T_MaxHigh, N)

    s_loop = []
    T_loop = []

    # saturated liquid (Q=0)
    for T in T_range:
        try:
            Ref.update(QT_INPUTS, 0, T)
            s_loop.append(Ref.smass())
            T_loop.append(T - KELVIN)
        except Exception:
            continue

    # critical point
    Ref.update(PT_INPUTS, P_crit, T_MaxHigh)
    s_crit = Ref.smass()
    s_loop.append(s_crit)
    T_loop.append(T_MaxHigh - KELVIN)

    # saturated vapor (Q=1)
    for T in reversed(T_range):
        try:
            Ref.update(QT_INPUTS, 1, T)
            s_loop.append(Ref.smass())
            T_loop.append(T - KELVIN)
        except Exception:
            continue

    # === Cycle points ===
    Pressure_cond = cycle.ref["P3"]
    Pressure_evap = cycle.ref["P2"]

    # get known points from cycle
    Ref.update(PQ_INPUTS, Pressure_evap, 1)
    s1, T1 = Ref.smass(), Ref.T()
    s2, T2 = cycle.ref["s2"], cycle.ref["T2"]
    s3, T3 = cycle.ref["s3"], cycle.ref["T3"]
    s4, h4, T4 = cycle.ref["s4"], cycle.ref["h4"], cycle.ref["T4"]
    s5, h5, T5 = cycle.ref["s5"], cycle.ref["h5"], cycle.ref["T5"]

    if IHX:
        s6, T6 = cycle.ref["s6"], cycle.ref["T6"]

    # choose correct expansion inlet depending on IHX
    if not IHX:
        h_valve = h4
        s_evap1 = np.linspace(s5, s1, N)
        s_points = [s1, s2, s3, s4, s5]
        T_points = [T1, T2, T3, T4, T5]
        P_points = [Pressure_evap, Pressure_evap, Pressure_cond, Pressure_cond, Pressure_evap]
    else:
        h_valve = h5
        s_evap1 = np.linspace(s6, s1, N)
        s_points = [s1, s2, s3, s4, s5, s6]
        T_points = [T1, T2, T3, T4, T5, T6]
        P_points = [Pressure_evap, Pressure_evap, Pressure_cond, Pressure_cond, Pressure_cond, Pressure_evap]

    # --- process lines ---
    T_evap1 = np.array(
        [Ref.update(PSmass_INPUTS, Pressure_evap, s) or Ref.T() for s in s_evap1]
    ) - KELVIN

    s_evap2 = np.linspace(s1, s2, N)
    T_evap2 = np.array(
        [Ref.update(PSmass_INPUTS, Pressure_evap, s) or Ref.T() for s in s_evap2]
    ) - KELVIN

    s_cond = np.linspace(s3, s4, N)
    T_cond = np.array(
        [Ref.update(PSmass_INPUTS, Pressure_cond, s) or Ref.T() for s in s_cond]
    ) - KELVIN

    # throttling line
    pressures = np.linspace(Pressure_cond, Pressure_evap, N)
    s_valve = np.empty(N)
    T_valve = np.empty(N)
    for i, p in enumerate(pressures):
        try:
            Ref.update(HmassP_INPUTS, h_valve, p)
            s_valve[i] = Ref.smass()
            T_valve[i] = Ref.T() - KELVIN
        except Exception:
            # leave as whatever was there; this only affects a few bad points
            s_valve[i] = np.nan
            T_valve[i] = np.nan

    if not IHX:
        s_combined = np.concatenate([s_evap1, s_evap2, s_cond, s_valve])
        T_combined = np.concatenate([T_evap1, T_evap2, T_cond, T_valve])
    else:
        # IHX hot side
        s_IHX_high = np.linspace(s4, s5, N)
        T_IHX_high = np.array(
            [Ref.update(PSmass_INPUTS, Pressure_cond, s) or Ref.T() for s in s_IHX_high]
        ) - KELVIN

        s_combined = np.concatenate([s_evap1, s_evap2, s_cond, s_IHX_high, s_valve])
        T_combined = np.concatenate([T_evap1, T_evap2, T_cond, T_IHX_high, T_valve])

    # --- plotting ---
    plt.plot(s_combined, T_combined, label="Heat pump cycle")
    plt.plot(s_loop, T_loop, "--", color="black", label="Saturation dome")

    if extended:
        # annotate points with T_i and P_i
        for i, (s, T, P) in enumerate(zip(s_points, T_points, P_points), start=1):
            ax.plot(s, T - KELVIN, "ro")
            fontsize = 20

            # different offsets per region to avoid overlap
            if i == 1:
                xoff, yoff_T, yoff_P = 10, 0, -20
                ha = "left"
            elif i < 4:
                xoff, yoff_T, yoff_P = 10, 0, -20
                ha = "left"
            elif i == 4 or (i == 5 and IHX):
                xoff, yoff_T, yoff_P = -10, 20, 0
                ha = "right"
            else:
                xoff, yoff_T, yoff_P = -10, 0, -20
                ha = "right"

            ax.annotate(
                f"$T_{i}$ = {T - KELVIN:.1f}",
                (s, T - KELVIN),
                xytext=(xoff, yoff_T),
                textcoords="offset points",
                fontsize=fontsize,
                color="red",
                ha=ha,
            )
            ax.annotate(
                f"$P_{i}$ = {P/1e5:.2f}",
                (s, T - KELVIN),
                xytext=(xoff, yoff_P),
                textcoords="offset points",
                fontsize=fontsize,
                color="red",
                ha=ha,
            )
    else:
        # minimal numbering
        for i, (s, T) in enumerate(zip(s_points, T_points), start=1):
            plt.plot(s, T - KELVIN, "ro")
            if i == 3:
                plt.text(s + 10, T - KELVIN, f"{i}", fontsize=22, color="red", ha="left", va="center")
            elif i == 6:
                plt.text(s - 10, T - KELVIN, f"{i}", fontsize=22, color="red", ha="right", va="center")
            else:
                plt.text(s - 7, T - KELVIN, f"{i}", fontsize=22, color="red", ha="right", va="bottom")

    # critical point
    plt.scatter(s_crit, T_MaxHigh - KELVIN, label="Critical point", color="black", marker="*", s=70)

    plt.xlabel("Entropy [J/kgK]")
    plt.ylabel("Temperature [°C]")
    plt.grid(True)
    plt.legend()

    # re-fit limits to annotations
    ax.relim()
    for t in ax.texts:
        ax.update_datalim(
            t.get_window_extent(renderer=plt.gcf().canvas.get_renderer()).transformed(ax.transData.inverted())
        )
    ax.autoscale_view()
    plt.tight_layout()
    plt.show()


def plot_TQ_diagram_all(cycle, dryer, plt_sink, plt_source, IHX, HTF):
    """
    Plot all T–Q curves (sink, source, refrigerant, IHX) on one figure.
    """
    def plot_TQ_curve(h_array, T_array, m_dot, label, color):
        Q = (h_array - h_array[0]) * m_dot / 1e3  # kJ
        T_C = T_array - KELVIN
        plt.plot(Q, T_C, label=label, lw=2, color=color)

    # condenser-side
    if plt_sink:
        plot_TQ_curve(np.flip(cycle.ref["h_Cond"]), np.flip(cycle.ref["T_Cond"]),
                      cycle.ref["m_dot"], "Condenser", "black")
        
        if HTF:
            plot_TQ_curve(cycle.htf["h_sink"], cycle.htf["T_sink"], 
                         cycle.htf["m_dot_sink"], "HTF condenser", "purple")
            
        plot_TQ_curve(cycle.air["h_sink"], cycle.air["T_sink"], dryer.m_da1, 
                      "Sink Air Stream", "red")

    # evaporator-side
    if plt_source:
        plot_TQ_curve(np.flip(cycle.air["h_source"]), np.flip(cycle.air["T_source"]),
                      dryer.m_da4, "Source Air Stream", "darkblue")
        
        if HTF:
            plot_TQ_curve(cycle.htf["h_source"], cycle.htf["T_source"], 
                          cycle.htf["m_dot_source"], "HTF evaporator", "orange")
            
        plot_TQ_curve(cycle.ref["h_Evap"], cycle.ref["T_Evap"], cycle.ref["m_dot"], 
                      "Evaporator", "green")

    # IHX
    if IHX:
        plot_TQ_curve(np.flip(cycle.ref["h_IHX_high"]), np.flip(cycle.ref["T_IHX_high"]),
                      cycle.ref["m_dot"], "IHX hot stream", "red")
        
        plot_TQ_curve(cycle.ref["h_IHX_low"], cycle.ref["T_IHX_low"], cycle.ref["m_dot"],
                      "IHX cold stream", "blue")

    plt.xlabel("Heat Transfer [kW]")
    plt.ylabel("Temperature [°C]")
    plt.grid(True)
    plt.legend(prop={"size": 17})
    plt.tight_layout()
    plt.show()
