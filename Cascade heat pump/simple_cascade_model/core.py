# -*- coding: utf-8 -*-
"""
Created on Wed May 21 12:24:54 2025

@author: JHLam
"""
import numpy as np
import math
from CoolProp.CoolProp import HAPropsSI
from CoolProp import (
    AbstractState,
    HmassP_INPUTS,
    PQ_INPUTS,
    QT_INPUTS,
    PSmass_INPUTS,
    PT_INPUTS,
    SmassT_INPUTS,
)
from .constants import (
    DISCRETIZATION,
    T_SUPERHEAT,
    P_AMB,
    MAX_ITER,
    TOLERANCE_ETAC,
    KELVIN,
    PENALTY,
)
from .utils import get_air_temp_profile, EntropyProduction
from scipy.optimize import minimize_scalar
from .constraints import (
    pinch_constraint_cond_upper,
    pinch_constraint_IHX_upper,
    pinch_constraint_shared_hex,
    pinch_constraint_IHX_lower,
    pinch_constraint_evap_lower,
)


def UpperCycle(RefHigh, cycle, dryer, T_PinchInternal, T_PinchAir, etac,
               IHX_upper, T_start_point, HTF):
    # sink air profile
    cycle.air["h_sink"] = np.linspace(dryer.pt1.H, dryer.pt2.H, DISCRETIZATION)
    cycle.air["T_sink"] = HAPropsSI("T", "H", cycle.air["h_sink"], "P", P_AMB, "W", dryer.pt1.W)

    # HTF on sink side
    if HTF:
        HtfUpper = AbstractState("INCOMP", "T66")
        T_in = cycle.air["T_sink"][0] + T_PinchAir
        HtfUpper.update(PT_INPUTS, P_AMB, T_in)
        h_in = HtfUpper.hmass()
        cycle.htf["s_sink_in"] = HtfUpper.smass()

        T_out = cycle.air["T_sink"][-1] + T_PinchAir
        HtfUpper.update(PT_INPUTS, P_AMB, T_out)
        h_out = HtfUpper.hmass()
        cycle.htf["s_sink_out"] = HtfUpper.smass()

        cycle.htf["h_sink"] = np.linspace(h_in, h_out, DISCRETIZATION)
        cycle.htf["T_sink"] = np.array(
            [HtfUpper.update(HmassP_INPUTS, h, P_AMB) or HtfUpper.T() for h in cycle.htf["h_sink"]]
        )
        cycle.htf["m_dot_sink"] = dryer.Q_heating / (h_out - h_in)
        T_CondIn = T_out + T_PinchInternal
    else:
        T_in = cycle.air["T_sink"][0]
        T_out = cycle.air["T_sink"][-1]
        T_CondIn = T_out + T_PinchAir

    # critical props
    P_crit = RefHigh.p_critical()
    T_crit = RefHigh.T_critical()
    RefHigh.update(PT_INPUTS, P_crit, T_crit)
    s_crit = RefHigh.smass()

    N = DISCRETIZATION
    T2_increased = False
    first_IHX = True

    # point 1 (evap outlet)
    RefHigh.update(QT_INPUTS, 1, T_start_point + KELVIN)
    cycle.high["s1"] = RefHigh.smass()
    cycle.high["h1"] = RefHigh.hmass()
    cycle.high["P1"] = RefHigh.p()
    cycle.high["T1"] = RefHigh.T()

    # point 2 (after superheat)
    RefHigh.update(PT_INPUTS, RefHigh.p(), RefHigh.T() + T_SUPERHEAT)
    cycle.high["s2"] = RefHigh.smass()
    cycle.high["h2"] = RefHigh.hmass()
    cycle.high["P2"] = RefHigh.p()
    cycle.high["T2"] = RefHigh.T()
    cycle.high["rho2"] = RefHigh.rhomass()

    for _ in range(MAX_ITER):
        wet_detected = False

        # --- point 3: compressor outlet / find P3 for given etac ---
        def objective_high(p):
            try:
                RefHigh.update(PSmass_INPUTS, p, cycle.high["s2"])
                h3_is = RefHigh.hmass()
                RefHigh.update(PT_INPUTS, p, T_CondIn)
                h3 = RefHigh.hmass()
                eta_calc = (h3_is - cycle.high["h2"]) / (h3 - cycle.high["h2"])
                return abs(eta_calc - etac)
            except Exception:
                return 1

        try:
            RefHigh.update(SmassT_INPUTS, cycle.high["s2"], T_CondIn)
            p_max = RefHigh.p()
        except Exception:
            p_max = 2 * P_crit

        res_high = minimize_scalar(
            objective_high,
            bounds=(cycle.high["P2"], p_max),
            method="bounded",
            options={"maxiter": MAX_ITER, "xatol": TOLERANCE_ETAC},
        )
        cycle.high["P3"] = res_high.x

        # real outlet state using isentropic enthalpy and given efficiency
        RefHigh.update(PSmass_INPUTS, cycle.high["P3"], cycle.high["s2"])
        h3_is = RefHigh.hmass()
        cycle.high["h3"] = cycle.high["h2"] + (h3_is - cycle.high["h2"]) / etac
        RefHigh.update(HmassP_INPUTS, cycle.high["h3"], cycle.high["P3"])
        cycle.high["s3"] = RefHigh.smass()
        cycle.high["T3"] = RefHigh.T()

        # --- wet compression check ---
        s_vals = np.linspace(cycle.high["s2"], cycle.high["s3"], N)
        T_vals = np.linspace(cycle.high["T2"], cycle.high["T3"], N)
        for s, T in zip(s_vals, T_vals):
            try:
                RefHigh.update(SmassT_INPUTS, s, T)
                Q = RefHigh.Q()
                if 0 < Q < 1:
                    wet_detected = True
                    break
            except ValueError:
                continue

        if wet_detected:
            cycle.high["T2"] += 1
            T2_increased = True
            RefHigh.update(PT_INPUTS, cycle.high["P2"], cycle.high["T2"])
            cycle.high["s2"] = RefHigh.smass()
            cycle.high["h2"] = RefHigh.hmass()
            cycle.high["T2"] = RefHigh.T()
            cycle.high["P2"] = RefHigh.p()
            continue

        # --- point 4: condenser outlet ---
        if HTF:
            RefHigh.update(PT_INPUTS, cycle.high["P3"], T_in + T_PinchInternal)
        else:
            RefHigh.update(PT_INPUTS, cycle.high["P3"], T_in + T_PinchAir)
        cycle.high["s4"] = RefHigh.smass()
        cycle.high["h4"] = RefHigh.hmass()
        cycle.high["P4"] = RefHigh.p()
        cycle.high["T4"] = RefHigh.T()

        # --- transcritical / phase check ---
        if cycle.high["P4"] > P_crit:
            if cycle.high["s4"] > s_crit:
                T_CondIn += 1
                continue
        else:
            RefHigh.update(PQ_INPUTS, cycle.high["P3"], 0)
            T_bubble = RefHigh.T()
            if cycle.high["T4"] > T_bubble:
                T_CondIn += 1
                continue

        # --- condenser pinch ---
        cycle.high["h_Cond"] = np.linspace(cycle.high["h3"], cycle.high["h4"], N)
        cycle.high["T_Cond"] = np.array(
            [RefHigh.update(HmassP_INPUTS, h, cycle.high["P3"]) or RefHigh.T() for h in cycle.high["h_Cond"]]
        )
        Delta_T_min = pinch_constraint_cond_upper(cycle, T_PinchAir, T_PinchInternal, HTF)
        if Delta_T_min < 0:
            T_CondIn += abs(Delta_T_min)
            continue

        if IHX_upper:
            # point 5 (IHX out)
            delta_h_IHX = cycle.high["h2"] - cycle.high["h1"]
            RefHigh.update(HmassP_INPUTS, cycle.high["h4"] - delta_h_IHX, cycle.high["P4"])
            cycle.high["s5"] = RefHigh.smass()
            cycle.high["h5"] = RefHigh.hmass()
            cycle.high["T5"] = RefHigh.T()
            cycle.high["P5"] = RefHigh.p()

            # IHX pinch
            cycle.high["h_IHX_high"] = np.linspace(cycle.high["h4"], cycle.high["h5"], N)
            cycle.high["T_IHX_high"] = np.array(
                [RefHigh.update(HmassP_INPUTS, h, cycle.high["P4"]) or RefHigh.T() for h in cycle.high["h_IHX_high"]]
            )
            cycle.high["h_IHX_low"] = np.linspace(cycle.high["h1"], cycle.high["h2"], N)
            cycle.high["T_IHX_low"] = np.array(
                [RefHigh.update(HmassP_INPUTS, h, cycle.high["P2"]) or RefHigh.T() for h in cycle.high["h_IHX_low"]]
            )
            Delta_T_IHX = pinch_constraint_IHX_upper(cycle, T_PinchInternal)

            if Delta_T_IHX < 0:
                if T2_increased:
                    IHX_upper = False
                    continue
                else:
                    cycle.high["T2"] -= 1
                    RefHigh.update(PT_INPUTS, cycle.high["P2"], cycle.high["T2"])
                    cycle.high["s2"] = RefHigh.smass()
                    cycle.high["h2"] = RefHigh.hmass()
                    cycle.high["T2"] = RefHigh.T()
                    cycle.high["P2"] = RefHigh.p()
                    continue
            elif Delta_T_IHX > 1:
                if first_IHX:
                    cycle.high["T2"] += float(math.floor(Delta_T_IHX)) - 2
                    first_IHX = False
                else:
                    cycle.high["T2"] += 1
                RefHigh.update(PT_INPUTS, cycle.high["P2"], cycle.high["T2"])
                cycle.high["s2"] = RefHigh.smass()
                cycle.high["h2"] = RefHigh.hmass()
                cycle.high["T2"] = RefHigh.T()
                cycle.high["P2"] = RefHigh.p()
                continue
            else:
                # point 6 (expansion)
                RefHigh.update(HmassP_INPUTS, cycle.high["h5"], cycle.high["P2"])
                cycle.high["s6"] = RefHigh.smass()
                cycle.high["h6"] = RefHigh.hmass()
                cycle.high["T6"] = RefHigh.T()
                cycle.high["P6"] = RefHigh.p()

                # shared HEX profile
                cycle.high["h_Evap"] = np.linspace(cycle.high["h6"], cycle.high["h1"], N)
                cycle.high["T_Evap"] = np.array(
                    [RefHigh.update(HmassP_INPUTS, h, cycle.high["P2"]) or RefHigh.T() for h in cycle.high["h_Evap"]]
                )
        else:
            # no IHX → expansion right away
            RefHigh.update(HmassP_INPUTS, cycle.high["h4"], cycle.high["P2"])
            cycle.high["s5"] = RefHigh.smass()
            cycle.high["h5"] = RefHigh.hmass()
            cycle.high["T5"] = RefHigh.T()
            cycle.high["P5"] = RefHigh.p()

            cycle.high["h_Evap"] = np.linspace(cycle.high["h5"], cycle.high["h2"], N)
            cycle.high["T_Evap"] = np.array(
                [RefHigh.update(HmassP_INPUTS, h, cycle.high["P2"]) or RefHigh.T() for h in cycle.high["h_Evap"]]
            )

        # constraints satisfied
        break
    else:
        raise RuntimeError("UpperCycle failed to converge within MAX_ITER.")

    # refrigerant mass flow upper cycle
    delta_h = cycle.high["h3"] - cycle.high["h4"]
    cycle.high["m_dot"] = dryer.Q_heating / delta_h

    return IHX_upper


def LowerCycle(RefLow, cycle, dryer, T_PinchInternal, T_PinchAir, etac,
               IHX_upper, IHX_lower, HTF):
    P_crit = RefLow.p_critical()

    N = DISCRETIZATION
    cond_iteration = 0
    T7_increased = False
    Optimisation_finished = False
    T_super_lowered = False
    best_sigma = PENALTY

    # Q from dryer 3 → 4
    Delta_Q_pt3_pt4_air = (dryer.pt3.H - dryer.pt4.H) * dryer.m_da4

    # initial condenser inlet temp
    if IHX_upper:
        T_CondIn = cycle.high["T1"] + T_PinchInternal
    else:
        T_CondIn = cycle.high["T2"] + T_PinchInternal

    T_super = T_SUPERHEAT

    if HTF:
        cycle.low["T6"] = dryer.pt4.T - T_PinchAir - T_PinchInternal + 1
        HtfLower = AbstractState("INCOMP", "DowQ")
    else:
        cycle.low["T6"] = dryer.pt4.T - T_PinchAir + 1

    for _ in range(MAX_ITER):
        wet_detected = False

        # point 6: evap outlet
        RefLow.update(QT_INPUTS, 1, cycle.low["T6"])
        cycle.low["s6"] = RefLow.smass()
        cycle.low["h6"] = RefLow.hmass()
        cycle.low["P6"] = RefLow.p()

        # point 7: after superheat
        RefLow.update(PT_INPUTS, RefLow.p(), cycle.low["T6"] + T_super)
        cycle.low["s7"] = RefLow.smass()
        cycle.low["h7"] = RefLow.hmass()
        cycle.low["P7"] = RefLow.p()
        cycle.low["T7"] = RefLow.T()
        cycle.low["rho7"] = RefLow.rhomass()

        # point 8: compressor outlet
        def objective_low(p):
            try:
                RefLow.update(PSmass_INPUTS, p, cycle.low["s7"])
                h8_is = RefLow.hmass()
                RefLow.update(PT_INPUTS, p, T_CondIn)
                h8 = RefLow.hmass()
                eta_calc = (h8_is - cycle.low["h7"]) / (h8 - cycle.low["h7"])
                return abs(eta_calc - etac)
            except Exception:
                return 1

        RefLow.update(SmassT_INPUTS, cycle.low["s7"], T_CondIn)
        p_max = RefLow.p()
        res_low = minimize_scalar(
            objective_low,
            bounds=(cycle.low["P6"], p_max),
            method="bounded",
            options={"maxiter": MAX_ITER, "xatol": TOLERANCE_ETAC},
        )
        cycle.low["P8"] = res_low.x
        RefLow.update(PT_INPUTS, cycle.low["P8"], T_CondIn)
        cycle.low["h8"] = RefLow.hmass()
        RefLow.update(HmassP_INPUTS, cycle.low["h8"], cycle.low["P8"])
        cycle.low["s8"] = RefLow.smass()
        cycle.low["T8"] = RefLow.T()

        # point 9: condenser outlet
        if IHX_upper:
            RefLow.update(PT_INPUTS, cycle.low["P8"], cycle.high["T6"] + T_PinchInternal)
        else:
            RefLow.update(PT_INPUTS, cycle.low["P8"], cycle.high["T5"] + T_PinchInternal)
        cycle.low["s9"] = RefLow.smass()
        cycle.low["h9"] = RefLow.hmass()
        cycle.low["P9"] = RefLow.p()
        cycle.low["T9"] = RefLow.T()

        # transcritical / phase check
        if cycle.low["P8"] > P_crit:
            raise RuntimeError("Optimization run is stopped because the lower cycle was transcritical")
        else:
            RefLow.update(PQ_INPUTS, cycle.low["P8"], 0)
            T_bubble = RefLow.T()
            if cycle.low["T9"] > T_bubble:
                T_CondIn += 1
                continue

        # wet compression check
        s_vals = np.linspace(cycle.low["s7"], cycle.low["s8"], N)
        T_vals = np.linspace(cycle.low["T7"], cycle.low["T8"], N)
        for s, T in zip(s_vals, T_vals):
            try:
                RefLow.update(SmassT_INPUTS, s, T)
                Q = RefLow.Q()
                if 0 < Q < 1:
                    wet_detected = True
                    break
            except ValueError:
                continue

        if wet_detected:
            T_super += 1
            T7_increased = True
            continue

        # mass flow lower cycle
        delta_h = cycle.low["h8"] - cycle.low["h9"]
        if IHX_upper:
            cycle.low["m_dot"] = ((cycle.high["h1"] - cycle.high["h6"]) * cycle.high["m_dot"]) / delta_h
        else:
            cycle.low["m_dot"] = ((cycle.high["h2"] - cycle.high["h5"]) * cycle.high["m_dot"]) / delta_h

        if IHX_lower:
            # point 10 (IHX)
            delta_h_IHX = cycle.low["h7"] - cycle.low["h6"]
            RefLow.update(HmassP_INPUTS, cycle.low["h9"] - delta_h_IHX, cycle.low["P9"])
            cycle.low["s10"] = RefLow.smass()
            cycle.low["h10"] = RefLow.hmass()
            cycle.low["T10"] = RefLow.T()
            cycle.low["P10"] = RefLow.p()

            # point 11 (expansion)
            RefLow.update(HmassP_INPUTS, cycle.low["h10"], cycle.low["P7"])
            cycle.low["s11"] = RefLow.smass()
            cycle.low["h11"] = RefLow.hmass()
            cycle.low["T11"] = RefLow.T()
            cycle.low["P11"] = RefLow.p()

            # source air profile
            h_air_out = dryer.pt3.H - ((cycle.low["h6"] - cycle.low["h11"]) * cycle.low["m_dot"]) / dryer.m_da4
            cycle.air["h_source"] = np.linspace(dryer.pt3.H, h_air_out, N)
            cycle.air["T_source"] = get_air_temp_profile(cycle.air["h_source"], 
                                            dryer.pt4.W, dryer.pt4.H, 1, P_AMB)
            cycle.air["s5"] = HAPropsSI("S", "H", cycle.air["h_source"][-1],"T",
                                        cycle.air["T_source"][-1], "P", P_AMB)

            if HTF:
                T_in = cycle.air["T_source"][-1] - T_PinchAir
                HtfLower.update(PT_INPUTS, P_AMB, T_in)
                h_in = HtfLower.hmass()
                cycle.htf["s_source_in"] = HtfLower.smass()

                T_median = dryer.pt4.T - T_PinchAir
                HtfLower.update(PT_INPUTS, P_AMB, T_median)
                h_median = HtfLower.hmass()

                Delta_Q_HTF = (cycle.low["h6"] - cycle.low["h11"]) * cycle.low["m_dot"] - Delta_Q_pt3_pt4_air
                cycle.htf["m_dot_source"] = Delta_Q_HTF / (h_median - h_in)

                h_out = h_median + Delta_Q_pt3_pt4_air / cycle.htf["m_dot_source"]
                HtfLower.update(HmassP_INPUTS, h_out, P_AMB)
                cycle.htf["s_source_out"] = HtfLower.smass()

                cycle.htf["h_source"] = np.linspace(h_in, h_out, DISCRETIZATION)
                cycle.htf["T_source"] = np.array(
                    [HtfLower.update(HmassP_INPUTS, h, P_AMB) or HtfLower.T() for h in cycle.htf["h_source"]]
                )

            # evaporator pinch
            cycle.low["h_Evap"] = np.linspace(cycle.low["h11"], cycle.low["h6"], N)
            cycle.low["T_Evap"] = np.array(
                [RefLow.update(HmassP_INPUTS, h, cycle.low["P7"]) or RefLow.T() for h in cycle.low["h_Evap"]]
            )
            Delta_T_min_evap = pinch_constraint_evap_lower(cycle, T_PinchAir, T_PinchInternal, HTF)
            if Delta_T_min_evap < 0:
                cycle.low["T6"] -= 1
                continue

            # condenser (shared HEX) pinch
            cycle.low["h_Cond"] = np.linspace(cycle.low["h8"], cycle.low["h9"], N)
            cycle.low["T_Cond"] = np.array(
                [RefLow.update(HmassP_INPUTS, h, cycle.low["P8"]) or RefLow.T() for h in cycle.low["h_Cond"]]
            )
            Delta_T_min_cond = pinch_constraint_shared_hex(cycle, T_PinchInternal)
            if Delta_T_min_cond < 0:
                cond_iteration += 1
                if cond_iteration <= 1:
                    T_CondIn += abs(Delta_T_min_cond)
                else:
                    T_CondIn += 1
                continue

            # lower IHX pinch
            cycle.low["h_IHX_high"] = np.linspace(cycle.low["h9"], cycle.low["h10"], N)
            cycle.low["T_IHX_high"] = np.array(
                [RefLow.update(HmassP_INPUTS, h, cycle.low["P9"]) or RefLow.T() for h in cycle.low["h_IHX_high"]]
            )
            cycle.low["h_IHX_low"] = np.linspace(cycle.low["h6"], cycle.low["h7"], N)
            cycle.low["T_IHX_low"] = np.array(
                [RefLow.update(HmassP_INPUTS, h, cycle.low["P6"]) or RefLow.T() for h in cycle.low["h_IHX_low"]]
            )
            Delta_T_IHX = pinch_constraint_IHX_lower(cycle, T_PinchInternal)

            if Delta_T_IHX < 0:
                if T7_increased:
                    IHX_lower = False
                    continue
                else:
                    T_super -= 1
                    T_super_lowered = True
                    continue
            elif not Optimisation_finished and not T_super_lowered and Delta_T_IHX > 1:
                sigma_now, _ = EntropyProduction(cycle, dryer, IHX_upper, IHX_lower, HTF, False)
                if sigma_now < best_sigma:
                    best_sigma = sigma_now
                    T_super += 1
                    continue
                else:
                    T_super -= 1
                    Optimisation_finished = True
                    continue

        else:
            # no lower IHX
            RefLow.update(HmassP_INPUTS, cycle.low["h9"], cycle.low["P7"])
            cycle.low["s10"] = RefLow.smass()
            cycle.low["h10"] = RefLow.hmass()
            cycle.low["T10"] = RefLow.T()

            # evaporator pinch
            cycle.low["h_Evap"] = np.linspace(cycle.low["h10"], cycle.low["h7"], N)
            cycle.low["T_Evap"] = np.array(
                [RefLow.update(HmassP_INPUTS, h, cycle.low["P7"]) or RefLow.T() for h in cycle.low["h_Evap"]]
            )

            # source air profile
            h_air_out = dryer.pt3.H - ((cycle.low["h7"] - cycle.low["h10"]) * cycle.low["m_dot"]) / dryer.m_da4
            cycle.air["h_source"] = np.linspace(dryer.pt3.H, h_air_out, N)
            cycle.air["T_source"] = get_air_temp_profile(
                cycle.air["h_source"], dryer.pt4.W, dryer.pt4.H, 1, P_AMB
            )
            cycle.air["s5"] = HAPropsSI(
                "S",
                "H",
                cycle.air["h_source"][-1],
                "T",
                cycle.air["T_source"][-1],
                "P",
                P_AMB,
            )

            if HTF:
                T_in = cycle.air["T_source"][-1] - T_PinchAir
                HtfLower.update(PT_INPUTS, P_AMB, T_in)
                h_in = HtfLower.hmass()
                cycle.htf["s_source_in"] = HtfLower.smass()

                T_median = dryer.pt4.T - T_PinchAir
                HtfLower.update(PT_INPUTS, P_AMB, T_median)
                h_median = HtfLower.hmass()

                Delta_Q_HTF = (cycle.low["h7"] - cycle.low["h10"]) * cycle.low["m_dot"] - Delta_Q_pt3_pt4_air
                cycle.htf["m_dot_source"] = Delta_Q_HTF / (h_median - h_in)

                h_out = h_median + Delta_Q_pt3_pt4_air / cycle.htf["m_dot_source"]
                HtfLower.update(HmassP_INPUTS, h_out, P_AMB)
                cycle.htf["s_source_out"] = HtfLower.smass()

                cycle.htf["h_source"] = np.linspace(h_in, h_out, DISCRETIZATION)
                cycle.htf["T_source"] = np.array(
                    [HtfLower.update(HmassP_INPUTS, h, P_AMB) or HtfLower.T() for h in cycle.htf["h_source"]]
                )

            Delta_T_min_evap = pinch_constraint_evap_lower(cycle, T_PinchAir, T_PinchInternal, HTF)
            if Delta_T_min_evap < 0:
                cycle.low["T6"] -= 1
                continue

            # condenser (shared HEX) pinch
            cycle.low["h_Cond"] = np.linspace(cycle.low["h8"], cycle.low["h9"], N)
            cycle.low["T_Cond"] = np.array(
                [RefLow.update(HmassP_INPUTS, h, cycle.low["P8"]) or RefLow.T() for h in cycle.low["h_Cond"]]
            )
            Delta_T_min_cond = pinch_constraint_shared_hex(cycle, T_PinchInternal)
            if Delta_T_min_cond < 0:
                cond_iteration += 1
                if cond_iteration <= 1:
                    T_CondIn += abs(Delta_T_min_cond)
                else:
                    T_CondIn += 1
                continue

        break
    else:
        raise RuntimeError("LowerCycle failed to converge within MAX_ITER.")

    return IHX_lower
