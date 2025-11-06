# -*- coding: utf-8 -*-
"""
Pinch-point helper functions for condenser, evaporator, and IHX.

These return:
    ΔT_min - T_pinch   (so negative means the pinch requirement is violated)
"""

import numpy as np


def pinch_cond(cycle, T_PinchAir, T_PinchInternal, HTF):
    """
    Condenser pinch:
    compare refrigerant condensation curve (high side, reversed) with
    either air sink or HTF sink.
    """
    T_CondHigh = cycle.ref.get("T_Cond")
    if T_CondHigh is None:
        return -1.0  # fail-safe

    # high-side profile should go from hot → cold to match sink direction
    T_CondHigh = np.flip(T_CondHigh)

    if HTF:
        T_sink = cycle.htf.get("T_sink")
        T_pinch = T_PinchInternal
    else:
        T_sink = cycle.air.get("T_sink")
        T_pinch = T_PinchAir

    if T_sink is None:
        return -1.0

    return np.min(T_CondHigh - T_sink) - T_pinch


def pinch_IHX(cycle, T_pinch):
    """
    IHX pinch:
    high-pressure side (reversed) vs low-pressure side.
    """
    T_IHX_high = cycle.ref.get("T_IHX_high")
    T_IHX_low = cycle.ref.get("T_IHX_low")

    if T_IHX_high is None or T_IHX_low is None:
        return -1.0

    T_IHX_high = np.flip(T_IHX_high)

    return np.min(T_IHX_high - T_IHX_low) - T_pinch


def pinch_evap(cycle, T_PinchAir, T_PinchInternal, HTF):
    """
    Evaporator pinch:
    refrigerant evaporation curve vs source (air or HTF).
    """
    T_EvapLow = cycle.ref.get("T_Evap")
    if T_EvapLow is None:
        return -1.0

    if HTF:
        T_source = cycle.htf.get("T_source")
        T_pinch = T_PinchInternal
    else:
        T_source = cycle.air.get("T_source")
        if T_source is None:
            return -1.0
        # for air we flip so hot source matches cold refrigerant
        T_source = np.flip(T_source)
        T_pinch = T_PinchAir

    if T_source is None:
        return -1.0

    return np.min(T_source - T_EvapLow) - T_pinch