# -*- coding: utf-8 -*-
"""
Created on Wed May 21 12:25:43 2025

@author: JHLam
"""

import numpy as np
from CoolProp.CoolProp import HAPropsSI


def get_air_temp_profile(h_array, W, H_limit, R, P_amb):
    """Return air temperature profile for a given enthalpy profile."""
    T_air = np.empty_like(h_array)
    for i, h in enumerate(h_array):
        if h >= H_limit:
            T_air[i] = HAPropsSI("T", "H", h, "P", P_amb, "W", W)
        else:
            T_air[i] = HAPropsSI("T", "H", h, "P", P_amb, "R", R)
    return T_air


def EntropyProduction(cycle, dryer, IHX, HTF, print_entropy_production, extended=True):
    """
    Compute entropy production of all relevant components.
    Returns (sigma_total, entropies_dict).
    """
    entropies = {}

    # Compressor
    entropies["sigma_Comp"] = (cycle.ref["s3"] - cycle.ref["s2"]) * cycle.ref["m_dot"]

    # Condenser + air/HTF high side
    if HTF:
        entropies["sigma_Cond"] = (
            (cycle.ref["s4"] - cycle.ref["s3"]) * cycle.ref["m_dot"]
            + (cycle.htf["s_sink_out"] - cycle.htf["s_sink_in"]) * cycle.htf["m_dot_sink"]
        )
        entropies["sigma_HTF_air_high"] = (
            (dryer.pt2.S - dryer.pt1.S) * dryer.m_da2
            + (cycle.htf["s_sink_in"] - cycle.htf["s_sink_out"]) * cycle.htf["m_dot_sink"]
        )
    else:
        entropies["sigma_Cond"] = (
            (cycle.ref["s4"] - cycle.ref["s3"]) * cycle.ref["m_dot"]
            + (dryer.pt2.S - dryer.pt1.S) * dryer.m_da2
        )
        entropies["sigma_HTF_air_high"] = 0

    # IHX / valve
    if IHX:
        entropies["sigma_IHX"] = (
            (cycle.ref["s5"] - cycle.ref["s4"]) * cycle.ref["m_dot"]
            + (cycle.ref["s2"] - cycle.ref["s1"]) * cycle.ref["m_dot"]
        )
        entropies["sigma_Valve"] = (cycle.ref["s6"] - cycle.ref["s5"]) * cycle.ref["m_dot"]
    else:
        entropies["sigma_IHX"] = 0
        entropies["sigma_Valve"] = (cycle.ref["s5"] - cycle.ref["s4"]) * cycle.ref["m_dot"]

    # Evaporator + air/HTF low side
    if HTF:
        if IHX:
            entropies["sigma_Evap"] = (
                (cycle.ref["s1"] - cycle.ref["s6"]) * cycle.ref["m_dot"]
                + (cycle.htf["s_source_in"] - cycle.htf["s_source_out"]) * cycle.htf["m_dot_source"]
            )
        else:
            entropies["sigma_Evap"] = (
                (cycle.ref["s2"] - cycle.ref["s5"]) * cycle.ref["m_dot"]
                + (cycle.htf["s_source_in"] - cycle.htf["s_source_out"]) * cycle.htf["m_dot_source"]
            )
        entropies["sigma_HTF_air_low"] = (
            (cycle.htf["s_source_out"] - cycle.htf["s_source_in"]) * cycle.htf["m_dot_source"]
            + (cycle.air["s5"] - dryer.pt3.S) * dryer.m_da4
        )
    else:
        if IHX:
            entropies["sigma_Evap"] = (
                (cycle.ref["s1"] - cycle.ref["s6"]) * cycle.ref["m_dot"]
                + (cycle.air["s5"] - dryer.pt3.S) * dryer.m_da4
            )
        else:
            entropies["sigma_Evap"] = (
                (cycle.ref["s2"] - cycle.ref["s5"]) * cycle.ref["m_dot"]
                + (cycle.air["s5"] - dryer.pt3.S) * dryer.m_da4
            )
        entropies["sigma_HTF_air_low"] = 0

    # Total entropy production
    sigma_total = (
        entropies["sigma_Comp"]
        + entropies["sigma_Cond"]
        + entropies["sigma_HTF_air_high"]
        + entropies["sigma_IHX"]
        + entropies["sigma_Valve"]
        + entropies["sigma_Evap"]
        + entropies["sigma_HTF_air_low"]
    )

    # Optional printing
    if print_entropy_production:
        print("\n--- Entropy production ---")
        print(f"Total entropy production:                   {sigma_total:.2f} W/K")
        print(f"Entropy production compressor:              {entropies['sigma_Comp']:.2f} W/K")
        if HTF:
            print(f"Entropy production condenser:               {entropies['sigma_Cond']:.2f} W/K")
            print(f"Entropy production HEX HTF to sink air:     {entropies['sigma_HTF_air_high']:.2f} W/K")
        else:
            print(f"Entropy production in condenser:            {entropies['sigma_Cond']:.2f} W/K")
        if IHX: 
            print(f"Entropy production IHX:                     {entropies['sigma_IHX']:.2f} W/K")
        print(f"Entropy production expansion valve:         {entropies['sigma_Valve']:.2f} W/K")
        if HTF:
            print(f"Entropy production HEX source air to HTF:   {entropies['sigma_HTF_air_low']:.2f} W/K")
            print(f"Entropy prodction evaporator:               {entropies['sigma_Evap']:.2f} W/K")
        else:
            print(f"Entropy prodction evaporator:               {entropies['sigma_Evap']:.2f} W/K")

    return sigma_total, entropies
