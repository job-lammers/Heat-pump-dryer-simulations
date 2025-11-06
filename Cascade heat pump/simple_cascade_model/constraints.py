# -*- coding: utf-8 -*-
"""
Created on Wed May 21 12:25:30 2025

@author: JHLam
"""
import numpy as np
    
def pinch_constraint_cond_upper(cycle, T_PinchAir, T_PinchInternal, HTF):
    T_CondHigh = np.flip(cycle.high.get("T_Cond"))
    if HTF:
        T_sink = cycle.htf.get("T_sink")
        T_pinch = T_PinchInternal
    else:
        T_sink = cycle.air.get("T_sink")
        T_pinch = T_PinchAir
    if T_CondHigh is None:
        return -1.0  # fail-safe fallback
    return np.min(T_CondHigh - T_sink) - T_pinch

def pinch_constraint_IHX_upper(cycle, T_pinch):
    T_IHX_high = np.flip(cycle.high.get("T_IHX_high"))
    T_IHX_low = cycle.high.get("T_IHX_low")
    if T_IHX_high is None or T_IHX_low is None:
        return -1.0  # fail-safe fallback
    return np.min(T_IHX_high - T_IHX_low) - T_pinch

def pinch_constraint_shared_hex(cycle, T_pinch):
    T_CondLow = np.flip(cycle.low.get("T_Cond"))
    T_EvapHigh = cycle.high.get("T_Evap")
    if T_CondLow is None or T_EvapHigh is None:
        return -1.0
    return np.min(T_CondLow - T_EvapHigh) - T_pinch

def pinch_constraint_IHX_lower(cycle, T_pinch):
    T_IHX_high = np.flip(cycle.low.get("T_IHX_high"))
    T_IHX_low = cycle.low.get("T_IHX_low")
    if T_IHX_high is None or T_IHX_low is None:
        return -1.0  # fail-safe fallback
    return np.min(T_IHX_high - T_IHX_low) - T_pinch

def pinch_constraint_evap_lower(cycle, T_PinchAir, T_PinchInternal, HTF):
    T_EvapLow = cycle.low.get("T_Evap")
    if HTF:
        T_source = cycle.htf.get("T_source")
        T_pinch = T_PinchInternal
    else:
        T_source = cycle.air.get("T_source")
        T_pinch = T_PinchAir
    if T_EvapLow is None or T_source is None:
        return -1.0
    return np.min(T_source - T_EvapLow) - T_pinch