# -*- coding: utf-8 -*-
"""
Utility functions for the cascade heat pump:
- air temperature reconstruction from h + (W or R)
- entropy production bookkeeping for all components

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


def EntropyProduction(cycle, dryer, IHX_upper, IHX_lower, HTF, print_entropy_production):
    """Compute and optionally print entropy production of each component."""
    ent = {}

    # --- upper compressor ---
    ent["sigma_CompHigh"] = (cycle.high["s3"] - cycle.high["s2"]) * cycle.high["m_dot"]

    # --- upper condenser + possibly HTF-to-air ---
    if HTF:
        ent["sigma_CondHigh"] = (cycle.high["s4"] - cycle.high["s3"]) * cycle.high["m_dot"] + \
                                (cycle.htf["s_sink_out"] - cycle.htf["s_sink_in"]) * cycle.htf["m_dot_sink"]
        ent["sigma_HTF_air_high"] = (dryer.pt2.S - dryer.pt1.S) * dryer.m_da2 + \
                                    (cycle.htf["s_sink_in"] - cycle.htf["s_sink_out"]) * cycle.htf["m_dot_sink"]
    else:
        ent["sigma_CondHigh"] = (cycle.high["s4"] - cycle.high["s3"]) * cycle.high["m_dot"] + \
                                (dryer.pt2.S - dryer.pt1.S) * dryer.m_da2
        ent["sigma_HTF_air_high"] = 0

    # --- upper IHX / valve / shared hex ---
    if IHX_upper:
        ent["sigma_IHX_upper"] = (cycle.high["s5"] - cycle.high["s4"]) * cycle.high["m_dot"] + \
                                 (cycle.high["s2"] - cycle.high["s1"]) * cycle.high["m_dot"]
        ent["sigma_ValveHigh"] = (cycle.high["s6"] - cycle.high["s5"]) * cycle.high["m_dot"]
        ent["sigma_SharedHex"] = (cycle.high["s1"] - cycle.high["s6"]) * cycle.high["m_dot"] + \
                                 (cycle.low["s9"] - cycle.low["s8"]) * cycle.low["m_dot"]
    else:
        ent["sigma_IHX_upper"] = 0
        ent["sigma_ValveHigh"] = (cycle.high["s5"] - cycle.high["s4"]) * cycle.high["m_dot"]
        ent["sigma_SharedHex"] = (cycle.high["s2"] - cycle.high["s5"]) * cycle.high["m_dot"] + \
                                 (cycle.low["s9"] - cycle.low["s8"]) * cycle.low["m_dot"]

    # --- lower compressor ---
    ent["sigma_CompLow"] = (cycle.low["s8"] - cycle.low["s7"]) * cycle.low["m_dot"]

    # --- lower IHX / valve ---
    if IHX_lower:
        ent["sigma_IHX_lower"] = (cycle.low["s10"] - cycle.low["s9"]) * cycle.low["m_dot"] + \
                                 (cycle.low["s7"] - cycle.low["s6"]) * cycle.low["m_dot"]
        ent["sigma_ValveLow"] = (cycle.low["s11"] - cycle.low["s10"]) * cycle.low["m_dot"]
    else:
        ent["sigma_IHX_lower"] = 0
        ent["sigma_ValveLow"] = (cycle.low["s10"] - cycle.low["s9"]) * cycle.low["m_dot"]

    # --- lower evaporator (air/HTF side) ---
    if HTF:
        if IHX_lower:
            ent["sigma_EvapLow"] = (cycle.low["s6"] - cycle.low["s11"]) * cycle.low["m_dot"] + \
                                   (cycle.htf["s_source_in"] - cycle.htf["s_source_out"]) * cycle.htf["m_dot_source"]
        else:
            ent["sigma_EvapLow"] = (cycle.low["s7"] - cycle.low["s10"]) * cycle.low["m_dot"] + \
                                   (cycle.htf["s_source_in"] - cycle.htf["s_source_out"]) * cycle.htf["m_dot_source"]
        ent["sigma_HTF_air_low"] = (cycle.htf["s_source_out"] - cycle.htf["s_source_in"]) * cycle.htf["m_dot_source"] + \
                                   (cycle.air["s5"] - dryer.pt3.S) * dryer.m_da4
    else:
        if IHX_lower:
            ent["sigma_EvapLow"] = (cycle.low["s6"] - cycle.low["s11"]) * cycle.low["m_dot"] + \
                                   (cycle.air["s5"] - dryer.pt3.S) * dryer.m_da4
        else:
            ent["sigma_EvapLow"] = (cycle.low["s7"] - cycle.low["s10"]) * cycle.low["m_dot"] + \
                                   (cycle.air["s5"] - dryer.pt3.S) * dryer.m_da4
        ent["sigma_HTF_air_low"] = 0

    sigma_total = (
        ent["sigma_CompHigh"] + ent["sigma_CondHigh"] + ent["sigma_HTF_air_high"] +
        ent["sigma_IHX_upper"] + ent["sigma_ValveHigh"] + ent["sigma_SharedHex"] +
        ent["sigma_CompLow"] + ent["sigma_IHX_lower"] + ent["sigma_ValveLow"] +
        ent["sigma_EvapLow"] + ent["sigma_HTF_air_low"]
    )

    # --- optional printout ---
    if print_entropy_production:
        print("\n--- Entropy production ---")
        print(f"Total entropy production:                  {sigma_total:.2f} W/K")
        print(f"\nEntropy production upper compressor:       {ent['sigma_CompHigh']:.2f} W/K")

        if HTF:
            print(f"Entropy production condenser:              {ent['sigma_CondHigh']:.2f} W/K")
            print(f"Entropy production HEX HTF → sink air:     {ent['sigma_HTF_air_high']:.2f} W/K")
        else:
            print(f"Entropy production condenser:              {ent['sigma_CondHigh']:.2f} W/K")

        if IHX_upper:
            print(f"Entropy production upper IHX:              {ent['sigma_IHX_upper']:.2f} W/K")
        print(f"Entropy production upper expansion valve:  {ent['sigma_ValveHigh']:.2f} W/K")
        
        print(f"\nEntropy production shared HEX:             {ent['sigma_SharedHex']:.2f} W/K")
        
        print(f"\nEntropy production lower compressor:       {ent['sigma_CompLow']:.2f} W/K")
        if IHX_lower:
            print(f"Entropy production lower IHX:              {ent['sigma_IHX_lower']:.2f} W/K")
        print(f"Entropy production lower expansion valve:  {ent['sigma_ValveLow']:.2f} W/K")

        if HTF:
            print(f"Entropy production HEX source air → HTF:   {ent['sigma_HTF_air_low']:.2f} W/K")
            print(f"Entropy production evaporator:             {ent['sigma_EvapLow']:.2f} W/K")
        else:
            print(f"Entropy production evaporator:             {ent['sigma_EvapLow']:.2f} W/K")

    return sigma_total, ent


