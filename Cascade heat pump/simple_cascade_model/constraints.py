# -*- coding: utf-8 -*-
"""
Pinch-point constraint functions for cascade heat pump cycles.

Each function checks the minimum temperature difference between
hot and cold streams for various heat exchangers.

@author: JHLam
"""
import numpy as np


# === Upper cycle constraints ===
def pinch_constraint_cond_upper(cycle, T_PinchAir, T_PinchInternal, HTF):
    """Condenser pinch in the upper cycle."""
    T_CondHigh = np.flip(cycle.high.get("T_Cond"))
    if HTF:
        T_sink, T_pinch = cycle.htf.get("T_sink"), T_PinchInternal
    else:
        T_sink, T_pinch = cycle.air.get("T_sink"), T_PinchAir
    if T_CondHigh is None or T_sink is None:
        return -1.0
    return np.min(T_CondHigh - T_sink) - T_pinch


def pinch_constraint_IHX_upper(cycle, T_pinch):
    """Internal heat exchanger pinch in upper cycle."""
    T_IHX_high = np.flip(cycle.high.get("T_IHX_high"))
    T_IHX_low = cycle.high.get("T_IHX_low")
    if T_IHX_high is None or T_IHX_low is None:
        return -1.0
    return np.min(T_IHX_high - T_IHX_low) - T_pinch


# === Shared HEX between upper and lower cycles ===
def pinch_constraint_shared_hex(cycle, T_pinch):
    """Pinch in shared heat exchanger between upper condenser and lower evaporator."""
    T_CondLow = np.flip(cycle.low.get("T_Cond"))
    T_EvapHigh = cycle.high.get("T_Evap")
    if T_CondLow is None or T_EvapHigh is None:
        return -1.0
    return np.min(T_CondLow - T_EvapHigh) - T_pinch


# === Lower cycle constraints ===
def pinch_constraint_IHX_lower(cycle, T_pinch):
    """Internal heat exchanger pinch in lower cycle."""
    T_IHX_high = np.flip(cycle.low.get("T_IHX_high"))
    T_IHX_low = cycle.low.get("T_IHX_low")
    if T_IHX_high is None or T_IHX_low is None:
        return -1.0
    return np.min(T_IHX_high - T_IHX_low) - T_pinch


def pinch_constraint_evap_lower(cycle, T_PinchAir, T_PinchInternal, HTF):
    """Evaporator pinch in lower cycle."""
    T_EvapLow = cycle.low.get("T_Evap")
    if HTF:
        T_source, T_pinch = cycle.htf.get("T_source"), T_PinchInternal
    else:
        T_source, T_pinch = cycle.air.get("T_source"), T_PinchAir
    if T_EvapLow is None or T_source is None:
        return -1.0
    return np.min(T_source - T_EvapLow) - T_pinch
