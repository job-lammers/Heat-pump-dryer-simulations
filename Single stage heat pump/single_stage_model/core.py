# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 10:23:08 2025

@author: JHLam
"""
import numpy as np
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
    PENALTY,
)
from .utils import get_air_temp_profile, EntropyProduction
from scipy.optimize import minimize_scalar
from .constraints import pinch_IHX, pinch_cond, pinch_evap


def HeatPumpCycle(Refrigerant, cycle, dryer, T_PinchInternal, T_PinchAir, etac, IHX, HTF):
    P_crit = Refrigerant.p_critical()
    T_crit = Refrigerant.T_critical()
    Refrigerant.update(PT_INPUTS, P_crit, T_crit)
    s_crit = Refrigerant.smass()

    N = DISCRETIZATION
    evap_iteration = 0
    ihx_iteration = 0
    T2_increased = False
    Optimisation_finished = False
    T_super_lowered = False
    best_sigma = PENALTY

    # Delta Q from point 3 to 4 of the drying air
    Delta_Q_pt3_pt4_air = (dryer.pt3.H - dryer.pt4.H) * dryer.m_da4

    cycle.air["h_sink"] = np.linspace(dryer.pt1.H, dryer.pt2.H, DISCRETIZATION)
    cycle.air["T_sink"] = HAPropsSI("T", "H", cycle.air["h_sink"], "P", P_AMB, "W", dryer.pt1.W)

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

        cycle.ref["T1"] = dryer.pt4.T - T_PinchAir - T_PinchInternal
        HtfLower = AbstractState("INCOMP", "DowQ")
    else:
        T_CondIn = cycle.air["T_sink"][-1] + T_PinchAir
        cycle.ref["T1"] = dryer.pt4.T - T_PinchAir

    T_super = T_SUPERHEAT

    for _ in range(MAX_ITER):
        wet_detected = False

        # --- Point 1 (evaporator outlet) ---
        Refrigerant.update(QT_INPUTS, 1, cycle.ref["T1"])
        cycle.ref["s1"] = Refrigerant.smass()
        cycle.ref["h1"] = Refrigerant.hmass()
        cycle.ref["P1"] = Refrigerant.p()

        # --- Point 2 (compressor inlet) ---
        Refrigerant.update(PT_INPUTS, Refrigerant.p(), cycle.ref["T1"] + T_super)
        cycle.ref["s2"] = Refrigerant.smass()
        cycle.ref["h2"] = Refrigerant.hmass()
        cycle.ref["P2"] = Refrigerant.p()
        cycle.ref["T2"] = Refrigerant.T()
        cycle.ref["rho2"] = Refrigerant.rhomass()

        # --- Point 3 (compressor outlet) ---
        # estimate condenser pressure

        def objective(p):
            try:
                Refrigerant.update(PSmass_INPUTS, p, cycle.ref["s2"])
                h3_is = Refrigerant.hmass()
                Refrigerant.update(PT_INPUTS, p, T_CondIn)
                h3 = Refrigerant.hmass()
                eta_calc = (h3_is - cycle.ref["h2"]) / (h3 - cycle.ref["h2"])
                return abs(eta_calc - etac)
            except Exception:
                return 1

        Refrigerant.update(SmassT_INPUTS, cycle.ref["s2"], T_CondIn)
        p_max = Refrigerant.p()
        res = minimize_scalar(
            objective,
            bounds=(cycle.ref["P1"], p_max),
            method="bounded",
            options={"maxiter": MAX_ITER, "xatol": TOLERANCE_ETAC},
        )
        cycle.ref["P3"] = res.x
        Refrigerant.update(PT_INPUTS, cycle.ref["P3"], T_CondIn)
        cycle.ref["h3"] = Refrigerant.hmass()
        Refrigerant.update(HmassP_INPUTS, cycle.ref["h3"], cycle.ref["P3"])
        cycle.ref["s3"] = Refrigerant.smass()
        cycle.ref["T3"] = Refrigerant.T()

        # --- Point 4 (condenser outlet) ---
        if HTF:
            Refrigerant.update(PT_INPUTS, cycle.ref["P3"], cycle.htf["T_sink"][0] + T_PinchInternal)
        else:
            Refrigerant.update(PT_INPUTS, cycle.ref["P3"], cycle.air["T_sink"][0] + T_PinchAir)
        cycle.ref["s4"] = Refrigerant.smass()
        cycle.ref["h4"] = Refrigerant.hmass()
        cycle.ref["P4"] = Refrigerant.p()
        cycle.ref["T4"] = Refrigerant.T()

        # --- Check transcritical and phase location ---
        if cycle.ref["P4"] > P_crit:
            if cycle.ref["s4"] > s_crit:
                T_CondIn += 1
                continue
        else:
            try:
                Refrigerant.update(PQ_INPUTS, cycle.ref["P3"], 0)
                T_bubble = Refrigerant.T()
                if cycle.ref["T4"] > T_bubble:
                    T_CondIn += 1
                    continue
            except Exception:
                T_CondIn += 1
                continue

        # --- Wet compression check ---
        s_vals = np.linspace(cycle.ref["s2"], cycle.ref["s3"], N)
        T_vals = np.linspace(cycle.ref["T2"], cycle.ref["T3"], N)
        for s, T in zip(s_vals, T_vals):
            try:
                Refrigerant.update(SmassT_INPUTS, s, T)
                Q = Refrigerant.Q()
                if 0 < Q < 1:
                    wet_detected = True
                    break
            except ValueError:
                continue

        if wet_detected:
            T_super += 1  # raise inlet temperature to avoid wet compression
            T2_increased = True
            continue

        # --- Mass flow refrigerant lower cycle ---
        delta_h = cycle.ref["h3"] - cycle.ref["h4"]
        cycle.ref["m_dot"] = dryer.Q_heating / delta_h

        if IHX:
            # --- point 5 (IHX out) ---
            delta_h_IHX = cycle.ref["h2"] - cycle.ref["h1"]
            Refrigerant.update(HmassP_INPUTS, cycle.ref["h4"] - delta_h_IHX, cycle.ref["P4"])
            cycle.ref["s5"] = Refrigerant.smass()
            cycle.ref["h5"] = Refrigerant.hmass()
            cycle.ref["T5"] = Refrigerant.T()
            cycle.ref["P5"] = Refrigerant.p()

            # --- Point 6 (Expansion valve out) ---
            Refrigerant.update(HmassP_INPUTS, cycle.ref["h5"], cycle.ref["P1"])
            cycle.ref["s6"] = Refrigerant.smass()
            cycle.ref["h6"] = Refrigerant.hmass()
            cycle.ref["T6"] = Refrigerant.T()
            cycle.ref["P6"] = Refrigerant.p()

            # --- Temperature profile source ---
            h_air_out = dryer.pt3.H - ((cycle.ref["h1"] - cycle.ref["h6"]) * cycle.ref["m_dot"]) / dryer.m_da4
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

                Delta_Q_HTF = (cycle.ref["h1"] - cycle.ref["h6"]) * cycle.ref["m_dot"] - Delta_Q_pt3_pt4_air
                cycle.htf["m_dot_source"] = Delta_Q_HTF / (h_median - h_in)

                h_out = h_median + Delta_Q_pt3_pt4_air / cycle.htf["m_dot_source"]
                HtfLower.update(HmassP_INPUTS, h_out, P_AMB)
                cycle.htf["s_source_out"] = HtfLower.smass()

                cycle.htf["h_source"] = np.linspace(h_in, h_out, DISCRETIZATION)
                cycle.htf["T_source"] = np.array(
                    [HtfLower.update(HmassP_INPUTS, h, P_AMB) or HtfLower.T() for h in cycle.htf["h_source"]]
                )

            # --- Pinch check in evaporator ---
            cycle.ref["h_Evap"] = np.linspace(cycle.ref["h6"], cycle.ref["h1"], N)
            cycle.ref["T_Evap"] = np.array(
                [Refrigerant.update(HmassP_INPUTS, h, cycle.ref["P2"]) or Refrigerant.T() for h in cycle.ref["h_Evap"]]
            )
            Delta_T_min_evap = pinch_evap(cycle, T_PinchAir, T_PinchInternal, HTF)
            if Delta_T_min_evap < 0:
                evap_iteration += 1
                if evap_iteration <= 1:
                    cycle.ref["T1"] -= abs(round(Delta_T_min_evap, 0))
                    continue
                else:
                    cycle.ref["T1"] -= 1
                    continue

            # --- Pinch check in condenser ---
            cycle.ref["h_Cond"] = np.linspace(cycle.ref["h3"], cycle.ref["h4"], N)
            cycle.ref["T_Cond"] = np.array(
                [Refrigerant.update(HmassP_INPUTS, h, cycle.ref["P3"]) or Refrigerant.T() for h in cycle.ref["h_Cond"]]
            )
            Delta_T_min_cond = pinch_cond(cycle, T_PinchAir, T_PinchInternal, HTF)

            if Delta_T_min_cond < 0:
                T_CondIn += np.ceil(abs(Delta_T_min_cond))
                continue

            # --- Pinch check in IHX ---
            cycle.ref["h_IHX_high"] = np.linspace(cycle.ref["h4"], cycle.ref["h5"], N)
            cycle.ref["T_IHX_high"] = np.array(
                [Refrigerant.update(HmassP_INPUTS, h, cycle.ref["P4"]) or Refrigerant.T() for h in cycle.ref["h_IHX_high"]]
            )

            cycle.ref["h_IHX_low"] = np.linspace(cycle.ref["h1"], cycle.ref["h2"], N)
            cycle.ref["T_IHX_low"] = np.array(
                [Refrigerant.update(HmassP_INPUTS, h, cycle.ref["P1"]) or Refrigerant.T() for h in cycle.ref["h_IHX_low"]]
            )
            Delta_T_IHX = pinch_IHX(cycle, T_PinchInternal)

            if Delta_T_IHX < 0:
                if T2_increased:
                    IHX = False
                    continue
                else:
                    T_super -= 1
                    T_super_lowered = True
                    continue
            elif not Optimisation_finished and not T_super_lowered and Delta_T_IHX > 1:
                sigma_now, _ = EntropyProduction(cycle, dryer, IHX, HTF, False)

                if sigma_now < best_sigma:
                    ihx_iteration += 1
                    best_sigma = sigma_now
                    if ihx_iteration <= 1:
                        T_super += abs(round(Delta_T_IHX, 0)) - 1
                    else:
                        T_super += 1
                    continue
                else:
                    T_super -= 1
                    Optimisation_finished = True
                    continue

        else:
            # --- Point 5 (Expansion valve out) ---
            Refrigerant.update(HmassP_INPUTS, cycle.ref["h4"], cycle.ref["P2"])
            cycle.ref["s5"] = Refrigerant.smass()
            cycle.ref["h5"] = Refrigerant.hmass()
            cycle.ref["T5"] = Refrigerant.T()
            cycle.ref["P5"] = Refrigerant.p()

            # --- Pinch check in evaporator ---
            cycle.ref["h_Evap"] = np.linspace(cycle.ref["h5"], cycle.ref["h2"], N)
            cycle.ref["T_Evap"] = np.array(
                [Refrigerant.update(HmassP_INPUTS, h, cycle.ref["P2"]) or Refrigerant.T() for h in cycle.ref["h_Evap"]]
            )

            # Temperature profile source
            h_air_out = dryer.pt3.H - ((cycle.ref["h2"] - cycle.ref["h5"]) * cycle.ref["m_dot"]) / dryer.m_da4
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

                Delta_Q_HTF = (cycle.ref["h2"] - cycle.ref["h5"]) * cycle.ref["m_dot"] - Delta_Q_pt3_pt4_air
                cycle.htf["m_dot_source"] = Delta_Q_HTF / (h_median - h_in)

                h_out = h_median + Delta_Q_pt3_pt4_air / cycle.htf["m_dot_source"]
                HtfLower.update(HmassP_INPUTS, h_out, P_AMB)
                cycle.htf["s_source_out"] = HtfLower.smass()

                cycle.htf["h_source"] = np.linspace(h_in, h_out, DISCRETIZATION)
                cycle.htf["T_source"] = np.array(
                    [HtfLower.update(HmassP_INPUTS, h, P_AMB) or HtfLower.T() for h in cycle.htf["h_source"]]
                )

            Delta_T_min_evap = pinch_evap(cycle, T_PinchAir, T_PinchInternal, HTF)

            if Delta_T_min_evap < 0:
                evap_iteration += 1
                if evap_iteration <= 1:
                    cycle.ref["T1"] -= abs(Delta_T_min_evap)
                    continue
                else:
                    cycle.ref["T1"] -= 1
                    continue

            # --- Pinch check in condenser ---
            cycle.ref["h_Cond"] = np.linspace(cycle.ref["h3"], cycle.ref["h4"], N)
            cycle.ref["T_Cond"] = np.array(
                [Refrigerant.update(HmassP_INPUTS, h, cycle.ref["P3"]) or Refrigerant.T() for h in cycle.ref["h_Cond"]]
            )
            Delta_T_min_cond = pinch_cond(cycle, T_PinchAir, T_PinchInternal, HTF)

            if Delta_T_min_cond < 0:
                T_CondIn += np.ceil(abs(Delta_T_min_cond))
                continue

        break

    else:
        # on non-convergence: keep last temperatures in cycle for inspection
        cycle.ref["h_Cond"] = np.linspace(cycle.ref["h3"], cycle.ref["h4"], N)
        cycle.ref["T_Cond"] = np.array(
            [Refrigerant.update(HmassP_INPUTS, h, cycle.ref["P3"]) or Refrigerant.T() for h in cycle.ref["h_Cond"]]
        )
        
    return IHX
