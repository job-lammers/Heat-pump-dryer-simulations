# -*- coding: utf-8 -*-
"""
Created on Wed May 21 12:25:43 2025

@author: JHLam
"""

import numpy as np
from CoolProp.CoolProp import HAPropsSI

def get_air_temp_profile(h_array, W, H_limit, R, P_amb):
    T_air = np.empty_like(h_array)
    for i, h in enumerate(h_array):
        if h >= H_limit:
            T_air[i] = HAPropsSI("T", "H", h, "P", P_amb, "W", W)
        else:
            T_air[i] = HAPropsSI("T", "H", h, "P", P_amb, "R", R)
    return T_air

def EntropyProduction(cycle, dryer, IHX_upper, IHX_lower, HTF, print_entropy_production):
    entropies = {}
    entropies["sigma_CompHigh"] = (cycle.high['s3'] - cycle.high['s2']) * cycle.high['m_dot']    
    if HTF: 
        entropies["sigma_CondHigh"] = (cycle.high['s4'] - cycle.high['s3']) * cycle.high['m_dot'] + \
            (cycle.htf['s_sink_out'] - cycle.htf['s_sink_in']) * cycle.htf['m_dot_sink'] 
        entropies["sigma_HTF_air_high"] = (dryer.pt2.S - dryer.pt1.S) * dryer.m_da2 + \
            (cycle.htf['s_sink_in'] - cycle.htf['s_sink_out']) * cycle.htf['m_dot_sink']
    else:
        entropies["sigma_CondHigh"] = (cycle.high['s4'] - cycle.high['s3']) * cycle.high['m_dot'] + \
                      (dryer.pt2.S - dryer.pt1.S) * dryer.m_da2
        entropies["sigma_HTF_air_high"] = 0
        
    if IHX_upper:
        entropies["sigma_IHX_upper"] = (cycle.high['s5'] - cycle.high['s4']) * cycle.high['m_dot'] + \
            (cycle.high['s2'] - cycle.high['s1']) * cycle.high['m_dot'] 
        entropies["sigma_ValveHigh"] = (cycle.high['s6'] - cycle.high['s5']) * cycle.high['m_dot']
        entropies["sigma_SharedHex"] = (cycle.high['s1'] - cycle.high['s6']) * cycle.high['m_dot'] + \
            (cycle.low['s9'] - cycle.low['s8'])*cycle.low['m_dot']
    else:
        entropies["sigma_IHX_upper"] = 0
        entropies["sigma_ValveHigh"] = (cycle.high['s5'] - cycle.high['s4']) * cycle.high['m_dot']
        entropies["sigma_SharedHex"] = (cycle.high['s2'] - cycle.high['s5']) * cycle.high['m_dot'] + \
            (cycle.low['s9'] - cycle.low['s8'])*cycle.low['m_dot']

    entropies["sigma_CompLow"] = (cycle.low['s8'] - cycle.low['s7']) * cycle.low['m_dot']
    
    if IHX_lower:
        entropies["sigma_IHX_lower"] = (cycle.low['s10'] - cycle.low['s9'])*cycle.low['m_dot'] + \
            (cycle.low['s7'] - cycle.low['s6'])*cycle.low['m_dot']
        entropies["sigma_ValveLow"] = (cycle.low['s11'] - cycle.low['s10']) * cycle.low['m_dot']
    else:
        entropies["sigma_IHX_lower"] = 0
        entropies["sigma_ValveLow"] = (cycle.low['s10'] - cycle.low['s9']) * cycle.low['m_dot']
    
    if HTF:
        if IHX_lower:
            entropies["sigma_EvapLow"] = (cycle.low['s6'] - cycle.low['s11']) * cycle.low['m_dot'] + \
                (cycle.htf['s_source_in'] - cycle.htf['s_source_out']) * cycle.htf["m_dot_source"]
        else:
            entropies["sigma_EvapLow"] = (cycle.low['s7'] - cycle.low['s10']) * cycle.low['m_dot'] + \
                (cycle.htf['s_source_in'] - cycle.htf['s_source_out']) * cycle.htf["m_dot_source"]
        entropies["sigma_HTF_air_low"] = (cycle.htf['s_source_out'] - cycle.htf['s_source_in']) * cycle.htf["m_dot_source"] + \
            (cycle.air['s5'] - dryer.pt3.S) * dryer.m_da4
    
    else:
        if IHX_lower:
            entropies["sigma_EvapLow"] = (cycle.low['s6'] - cycle.low['s11']) * cycle.low['m_dot'] + \
                (cycle.air['s5'] - dryer.pt3.S) * dryer.m_da4
        else:
            entropies["sigma_EvapLow"] = (cycle.low['s7'] - cycle.low['s10']) * cycle.low['m_dot'] + \
                (cycle.air['s5'] - dryer.pt3.S) * dryer.m_da4
        entropies["sigma_HTF_air_low"] = 0
        
    sigma_total = (entropies["sigma_CompHigh"] + entropies["sigma_CondHigh"] + entropies["sigma_HTF_air_high"] +
                   entropies["sigma_IHX_upper"] + entropies["sigma_ValveHigh"] +entropies["sigma_SharedHex"] +
                   entropies["sigma_CompLow"] + entropies["sigma_IHX_lower"] + entropies["sigma_ValveLow"] +
                   entropies["sigma_EvapLow"] + entropies["sigma_HTF_air_low"])
    
    # if print_entropy_production:    
        # print(f'The total entropy production is {sigma_total:.2f} W/K')
        # print(f'The entropy production in the upper compressor is {sigma_CompHigh :.2f} W/K')
        # if HTF:
        #     print(f'The entropy production in the HEX from wf to HTF in the upper cycle is {sigma_CondHigh :.2f} W/K')
        #     print(f'The entropy production in the HEX from HTF to air in the upper cycle is {sigma_HTF_air_high :.2f} W/K')
        # else: 
        #     print(f'The entropy production in the upper condenser is {sigma_CondHigh :.2f} W/K')
        # if IHX_upper:
        #     print(f'The entropy production in the upper IHX is {sigma_IHX_upper :.2f} W/K')
        # print(f'The entropy production in the upper expansion valve is {sigma_ValveHigh :.2f} W/K')
        # print(f'The entropy production in the shared HEX is {sigma_SharedHex :.2f} W/K')
        # print(f'The entropy production in the lower compressor is {sigma_CompLow :.2f} W/K')
        # if IHX_lower:
        #     print(f'The entropy production in the lower IHX is {sigma_IHX_lower :.2f} W/K')
        # print(f'The entropy production in the lower expansion valve is {sigma_ValveLow :.2f} W/K')
        # if HTF:
        #     print(f'The entropy production in the HEX from air to HTF in the lower cycle is {sigma_HTF_air_low :.2f} W/K')
        #     print(f'The entropy production in the HEX from HTF to wf in the lower cycle is {sigma_EvapLow :.2f} W/K')
        # else:
        #     print(f'The entropy production in the lower evaporator is {sigma_EvapLow :.2f} W/K')
    return sigma_total, entropies


def EntropyProductionUpper(cycle, dryer, IHX_upper, HTF):
    sigma_CompHigh = (cycle.high['s3'] - cycle.high['s2']) * cycle.high['m_dot']    
    if HTF: 
        sigma_CondHigh = (cycle.high['s4'] - cycle.high['s3']) * cycle.high['m_dot'] + \
            (cycle.htf['s_sink_out'] - cycle.htf['s_sink_in']) * cycle.htf['m_dot_sink'] 
        sigma_HTF_air_high = (dryer.pt2.S - dryer.pt1.S) * dryer.m_da2 + \
            (cycle.htf['s_sink_in'] - cycle.htf['s_sink_out']) * cycle.htf['m_dot_sink']
    else:
        sigma_CondHigh = (cycle.high['s4'] - cycle.high['s3']) * cycle.high['m_dot'] + \
                      (dryer.pt2.S - dryer.pt1.S) * dryer.m_da2
        sigma_HTF_air_high = 0
        
    if IHX_upper:
        sigma_IHX_upper = (cycle.high['s5'] - cycle.high['s4']) * cycle.high['m_dot'] + \
            (cycle.high['s2'] - cycle.high['s1']) * cycle.high['m_dot'] 
        sigma_ValveHigh = (cycle.high['s6'] - cycle.high['s5']) * cycle.high['m_dot']
    else:
        sigma_IHX_upper = 0
        sigma_ValveHigh = (cycle.high['s5'] - cycle.high['s4']) * cycle.high['m_dot']
    
    sigma_upper = sigma_CompHigh + sigma_CondHigh + sigma_HTF_air_high + sigma_IHX_upper + sigma_ValveHigh
    
    return sigma_upper
    
    
    