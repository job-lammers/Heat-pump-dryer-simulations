# -*- coding: utf-8 -*-
"""
Created on Thu May 22 13:19:12 2025

@author: JHLam
"""
import matplotlib.pyplot as plt
import numpy as np
from .constants import DISCRETIZATION, KELVIN
from CoolProp import QT_INPUTS, PQ_INPUTS, PSmass_INPUTS, HmassP_INPUTS, PT_INPUTS

def plot_Ts_diagram_with_dome(cycle, Ref, stage, IHX, IHX_high, extended=True):
    N = 100
    ax = plt.gca()
    
    # === Saturation dome refrigerant ===
    T_MinHigh = 40+273.15  # Kelvin
    T_MaxHigh = Ref.T_critical()
    P_crit = Ref.p_critical()
    T_range = np.linspace(T_MinHigh, T_MaxHigh, N)

    s_loop = []
    T_loop = []

    # Saturated liquid (Q=0), low T → high T
    for T in T_range:
        try:
            Ref.update(QT_INPUTS, 0, T)
            s_loop.append(Ref.smass())
            T_loop.append(T - KELVIN)
        except:
            continue
    
    Ref.update(PT_INPUTS, P_crit, T_MaxHigh)
    s_crit = Ref.smass()
    s_loop.append(Ref.smass())
    T_loop.append(T_MaxHigh - KELVIN)
    
    # Saturated vapor (Q=1), high T → low T
    for T in reversed(T_range):
        try:
            Ref.update(QT_INPUTS, 1, T)
            s_loop.append(Ref.smass())
            T_loop.append(T - KELVIN)
        except:
            continue

    # === Get state points s1–s5 and T1–T5 ===
    if stage == "Upper":
        Pressure_cond = cycle.high["P3"]
        Pressure_evap = cycle.high["P2"]
        
        Ref.update(PQ_INPUTS, Pressure_evap, 1)
        s1, T1 = Ref.smass(), Ref.T()
        s2, T2 = cycle.high["s2"], cycle.high["T2"]
        s3, T3 = cycle.high["s3"], cycle.high["T3"]
        s4, h4, T4 = cycle.high["s4"], cycle.high["h4"], cycle.high['T4']
        s5, h5, T5 = cycle.high["s5"], cycle.high["h5"], cycle.high["T5"]
        
        if IHX == True:
            s6, T6 = cycle.high["s6"], cycle.high["T6"]
    else:
        Pressure_cond = cycle.low["P8"]
        Pressure_evap = cycle.low["P7"]
        Ref.update(PQ_INPUTS, Pressure_evap, 1)
        s1, T1 = Ref.smass(), Ref.T()
        s2, T2 = cycle.low["s7"], cycle.low["T7"]
        s3, T3 = cycle.low["s8"], cycle.low["T8"]
        s4, h4, T4 = cycle.low["s9"], cycle.low["h9"], cycle.low['T9']
        s5, h5, T5 = cycle.low["s10"], cycle.low["h10"], cycle.low["T10"]
        
        if IHX == True:
            s6, T6 = cycle.low["s11"], cycle.low["T11"]

    # === Set the right parameters dependingon IHX or not ===
    if IHX == False:
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
        
    # === Plot the process lines ===
    T_evap1 = np.array([Ref.update(PSmass_INPUTS, Pressure_evap, s) or Ref.T() for s in s_evap1]) - KELVIN

    s_evap2 = np.linspace(s1, s2, N)
    T_evap2 = np.array([Ref.update(PSmass_INPUTS, Pressure_evap, s) or Ref.T() for s in s_evap2]) - KELVIN

    s_cond = np.linspace(s3, s4, N)
    T_cond = np.array([Ref.update(PSmass_INPUTS, Pressure_cond, s) or Ref.T() for s in s_cond]) - KELVIN

    pressures = np.linspace(Pressure_cond, Pressure_evap, N)
    s_valve = np.empty(N)
    T_valve = np.empty(N)
    
    for i, p in enumerate(pressures):
        try:
            Ref.update(HmassP_INPUTS, h_valve, p)
            s_valve[i] = Ref.smass()
            T_valve[i] = Ref.T() - KELVIN
        except:
            continue  # skip bad points
    
    if IHX == False:
        s_combined = np.concatenate([s_evap1, s_evap2, s_cond, s_valve])
        T_combined = np.concatenate([T_evap1, T_evap2, T_cond,  T_valve])
    else:
        s_IHX_high = np.linspace(s4, s5, N)
        T_IHX_high = np.array([Ref.update(PSmass_INPUTS, Pressure_cond, s) or Ref.T() for s in s_IHX_high]) - KELVIN
        
        s_combined = np.concatenate([s_evap1, s_evap2, s_cond, s_IHX_high, s_valve])
        T_combined = np.concatenate([T_evap1, T_evap2, T_cond, T_IHX_high, T_valve])
        
    plt.plot(s_combined, T_combined, label=f'{stage} cycle')
    plt.plot(s_loop, T_loop, '--', color="black", label="Saturation dome")
    
    if stage == "Upper":
        start=1
    elif IHX_high:
        start=7
    else:
        start=6
        
    if extended:     
        for i, (s, T, P) in enumerate(zip(s_points, T_points, P_points), start=start):
            ax.plot(s, T - KELVIN, 'ro')
            fontsize = 20
            count = i - start + 1
            if count == 1:
                # eerst T label
                ax.annotate(f'$T_{{{i}}}$ = {T - KELVIN:.1f}',
                            (s, T - KELVIN),
                            xytext=(10, 0),   # 10 punten naar rechts
                            textcoords="offset points",
                            fontsize=fontsize, color="red", ha='left', va='top')
                
                # dan P label, 25 punten onder T label
                ax.annotate(f'$P_{{{i}}}$ = {P/1e5:.2f}',
                            (s, T - KELVIN),
                            xytext=(10, -20),  # zelfde x, maar lager
                            textcoords="offset points",
                            fontsize=fontsize, color="red", ha='left', va='top')
            elif count < 4:
                # eerst T label
                ax.annotate(f'$T_{{{i}}}$ = {T - KELVIN:.1f}',
                            (s, T - KELVIN),
                            xytext=(10, 0),   # 10 punten naar rechts
                            textcoords="offset points",
                            fontsize=fontsize, color="red", ha='left', va='bottom')
                
                # dan P label, 25 punten onder T label
                ax.annotate(f'$P_{{{i}}}$ = {P/1e5:.2f}',
                            (s, T - KELVIN),
                            xytext=(10, -20),  # zelfde x, maar lager
                            textcoords="offset points",
                            fontsize=fontsize, color="red", ha='left', va='bottom')
            elif count == 4:
                # eerst T label
                ax.annotate(f'$T_{{{i}}}$ = {T - KELVIN:.1f}',
                            (s, T - KELVIN),
                            xytext=(-10, 20),   # 10 punten naar rechts
                            textcoords="offset points",
                            fontsize=fontsize, color="red", ha='right', va='bottom')
                
                # dan P label, 25 punten onder T label
                ax.annotate(f'$P_{{{i}}}$ = {P/1e5:.2f}',
                            (s, T - KELVIN),
                            xytext=(-10, 0),  # zelfde x, maar lager
                            textcoords="offset points",
                            fontsize=fontsize, color="red", ha='right', va='bottom')
            elif count== 5 and IHX:
                # eerst T label
                ax.annotate(f'$T_{{{i}}}$ = {T - KELVIN:.1f}',
                            (s, T - KELVIN),
                            xytext=(-10, 0),   # 10 punten naar rechts
                            textcoords="offset points",
                            fontsize=fontsize, color="red", ha='right', va='bottom')
                
                # dan P label, 25 punten onder T label
                ax.annotate(f'$P_{{{i}}}$ = {P/1e5:.2f}',
                            (s, T - KELVIN),
                            xytext=(-10, -20),  # zelfde x, maar lager
                            textcoords="offset points",
                            fontsize=fontsize, color="red", ha='right', va='bottom')
            else:
                # eerst T label
                ax.annotate(f'$T_{{{i}}}$ = {T - KELVIN:.1f}',
                            (s, T - KELVIN),
                            xytext=(-10, -5),   # 10 punten naar rechts
                            textcoords="offset points",
                            fontsize=fontsize, color="red", ha='left', va='top')
                
                # dan P label, 25 punten onder T label
                ax.annotate(f'$P_{{{i}}}$ = {P/1e5:.2f}',
                            (s, T - KELVIN),
                            xytext=(-10, -25),  # zelfde x, maar lager
                            textcoords="offset points",
                            fontsize=fontsize, color="red", ha='left', va='top')
    else:
        # === Plot the state points ===
        for i, (s, T) in enumerate(zip(s_points, T_points), start=start):
            plt.plot(s, T - KELVIN, 'ro')  # convert T to Celsius for plotting
            plt.text(s - 5, T - KELVIN, f'{i}', fontsize=20, color='red', ha='right', va='bottom')

        
    # === Plot critical point ===
    plt.scatter(s_crit, T_MaxHigh - KELVIN , label = "Critical point", color="black", marker = "*", s = 70)
    
    # === Labels and formatting ===
    plt.xlabel('Entropy [J/kgK]')
    plt.ylabel('Temperature [°C]')
    plt.grid(True)
    plt.legend()
    
    ax.relim()             # opnieuw limieten berekenen
    for t in ax.texts:  
        ax.update_datalim(t.get_window_extent(renderer=plt.gcf().canvas.get_renderer()).transformed(ax.transData.inverted()))
    ax.autoscale_view()
    plt.tight_layout()
    plt.show()

def plot_TQ_diagram_all(cycle, dryer, plt_sink, plt_shared, plt_source, IHX_high, IHX_low, HTF):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    def plot_TQ_curve(h_array, T_array, m_dot, label, color):
        Q = (h_array - h_array[0]) * m_dot / 1e3  # Energy in kJ
        T_C = T_array - KELVIN                    # Convert to °C
        plt.plot(Q, T_C, label=label, lw=2, color=color)

    # Plot each stream with correct mass flow rate
    if plt_sink:
        plot_TQ_curve(np.flip(cycle.high["h_Cond"]), np.flip(cycle.high["T_Cond"]), cycle.high["m_dot"], 'Condenser high cycle', 'black')
        if HTF:
            plot_TQ_curve(cycle.htf["h_sink"], cycle.htf["T_sink"], cycle.htf["m_dot_sink"], 'Heat transfer fluid', 'purple')
        plot_TQ_curve(cycle.air["h_sink"], cycle.air["T_sink"], dryer.m_da1, 'Sink Air Stream', 'red')
    if plt_shared:
        plot_TQ_curve(np.flip(cycle.low["h_Cond"]), np.flip(cycle.low["T_Cond"]), cycle.low["m_dot"], 'Condenser low cycle', 'blue')
        plot_TQ_curve(cycle.high["h_Evap"], cycle.high["T_Evap"], cycle.high["m_dot"], 'Evaporator high cycle', 'darkred')
    if plt_source:
        plot_TQ_curve(np.flip(cycle.air["h_source"]), np.flip(cycle.air["T_source"]), dryer.m_da4, 'Source Air Stream', 'darkblue')
        if HTF:
            plot_TQ_curve(cycle.htf["h_source"], cycle.htf["T_source"], cycle.htf["m_dot_source"], 'Heat transfer fluid', 'orange')
        plot_TQ_curve(cycle.low["h_Evap"], cycle.low["T_Evap"], cycle.low["m_dot"], 'Evaporator low cycle', 'green')
    if IHX_high:
        plot_TQ_curve(np.flip(cycle.high["h_IHX_high"]), np.flip(cycle.high["T_IHX_high"]), cycle.high["m_dot"], 'IHX high cycle hot fluid', 'red')
        plot_TQ_curve(cycle.high["h_IHX_low"], cycle.high["T_IHX_low"], cycle.high["m_dot"], 'IHX high cycle cold fluid', 'blue')
    if IHX_low:
        plot_TQ_curve(np.flip(cycle.low["h_IHX_high"]), np.flip(cycle.low["T_IHX_high"]), cycle.low["m_dot"], 'IHX low cycle hot fluid', 'red')
        plot_TQ_curve(cycle.low["h_IHX_low"], cycle.low["T_IHX_low"], cycle.low["m_dot"], 'IHX low cycle cold fluid', 'blue')
        
    plt.xlabel("Heat Transfer [kW]")
    plt.ylabel("Temperature [°C]")
    # plt.title("T–Q Diagram for All Heat Exchanger Streams")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_Ts_diagram_from_entropy_temperature(S1, T1, label1, m_dot_ev, S2, T2, label2, m_dot_cd):
    """
    Plot a T-s diagram for two air streams using given entropy and temperature values.

    Parameters:
    -----------
    S1, T1 : array-like
        Specific entropy [J/kg·K] and temperature [K] of stream 1.
    S2, T2 : array-like
        Specific entropy [J/kg·K] and temperature [K] of stream 2.
    label1, label2 : str
        Labels for the two streams.
    """

    # Convert entropy to W/K and temperature to °C for plotting
    S1_kJ = (np.array(S1)-S1[-1]) * m_dot_ev
    T1_C = np.array(T1) - 273.15
    S2_kJ = (np.array(S2)-S2[0]) * m_dot_cd
    T2_C = np.array(T2) - 273.15 

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(S2_kJ, T2_C, label=label2, lw=2)
    plt.plot(S1_kJ, T1_C, label=label1, lw=2)

    plt.xlabel("Entropy [W/K]")
    plt.ylabel("Temperature [°C]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()