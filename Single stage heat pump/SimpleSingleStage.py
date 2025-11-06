# -*- coding: utf-8 -*-
"""
Cascade heat pump model wrapper.

@author: JHLam
"""
from CoolProp.CoolProp import HAPropsSI
from CoolProp import AbstractState as AS
from simple_cascade_model.constants import P_AMB
from simple_cascade_model.utils import EntropyProduction
from simple_cascade_model.core import UpperCycle, LowerCycle
from simple_cascade_model.cyclestate import CycleState
from simple_cascade_model.plots import plot_Ts_diagram_with_dome, plot_TQ_diagram_all


class SimpleCascade:
    def __init__(self, mixture_low, mole_fractions_low, mixture_high, mole_fractions_high,
                 T_PinchInternal, T_PinchAir, dryer, etac, T_start_point,
                 IHX_upper, IHX_lower, plot, HTF, ExtendedCalculations):
        self.mixture_low = mixture_low
        self.mole_fractions_low = mole_fractions_low
        self.mixture_high = mixture_high
        self.mole_fractions_high = mole_fractions_high
        self.dryer = dryer
        self.T_PinchInternal = T_PinchInternal
        self.T_PinchAir = T_PinchAir
        self.etac = etac
        self.T_start_point = T_start_point
        self.IHX_upper = IHX_upper
        self.IHX_lower = IHX_lower
        self.plot = plot
        self.HTF = HTF
        self.ExtendedCalculations = ExtendedCalculations

        self.optimize_cycle()

    def optimize_cycle(self):
        # --- Refrigerant mixture setup ---
        RefHigh = AS("REFPROP", "&".join(self.mixture_high)); RefHigh.set_mole_fractions(self.mole_fractions_high)
        RefLow  = AS("REFPROP", "&".join(self.mixture_low));  RefLow.set_mole_fractions(self.mole_fractions_low)
        cycle = CycleState()

        # --- Upper (high-stage) cycle ---
        self.IHX_upper = UpperCycle(RefHigh, cycle, self.dryer, self.T_PinchInternal, self.T_PinchAir,
                                    self.etac, self.IHX_upper, self.T_start_point, self.HTF)

        if self.plot:
            plot_Ts_diagram_with_dome(cycle, RefHigh, "Upper", self.IHX_upper, self.IHX_upper)

        # --- Lower (low-stage) cycle ---
        self.IHX_lower = LowerCycle(RefLow, cycle, self.dryer, self.T_PinchInternal, self.T_PinchAir,
                                    self.etac, self.IHX_upper, self.IHX_lower, self.HTF)

        # --- Optional plots ---
        if self.plot:
            # Lower cycle
            plot_Ts_diagram_with_dome(cycle, RefLow, "Lower", self.IHX_lower, self.IHX_upper)
            # Source and sink
            plot_TQ_diagram_all(cycle, self.dryer, plt_sink=True, plt_shared=False, plt_source=True,
                                IHX_high=False, IHX_low=False, HTF=self.HTF)
            # Shared HEX
            plot_TQ_diagram_all(cycle, self.dryer, plt_sink=False, plt_shared=True, plt_source=False,
                                IHX_high=False, IHX_low=False, HTF=False)
            # IHX plots
            if self.IHX_upper:
                plot_TQ_diagram_all(cycle, self.dryer, plt_sink=False, plt_shared=False, plt_source=False,
                                    IHX_high=True, IHX_low=False, HTF=False)
            if self.IHX_lower:
                plot_TQ_diagram_all(cycle, self.dryer, plt_sink=False, plt_shared=False, plt_source=False,
                                    IHX_high=False, IHX_low=True, HTF=False)

        # --- Performance indicators ---
        Q_out = (cycle.high["h3"] - cycle.high["h4"]) * cycle.high["m_dot"]
        self.W_upper = (cycle.high["h3"] - cycle.high["h2"]) * cycle.high["m_dot"]
        self.W_lower = (cycle.low["h8"] - cycle.low["h7"]) * cycle.low["m_dot"]
        self.COP = Q_out / (self.W_upper + self.W_lower)

        self.sigma_total, self.entropies = EntropyProduction(cycle, self.dryer, self.IHX_upper,
                                                             self.IHX_lower, self.HTF, print_entropy_production=False)

        # --- Store cycle and key pressures ---
        self.cycle = cycle
        self.PR_upper = cycle.high["P3"] / cycle.high["P2"]; self.P_u_max = cycle.high["P3"]
        self.PR_lower = cycle.low["P8"] / cycle.low["P7"];  self.P_l_max = cycle.low["P8"]

        # --- Extended thermodynamic indicators ---
        if self.ExtendedCalculations:
            # Average source temperature (Lorenz)
            h_ev, T_ev = cycle.air["h_source"], cycle.air["T_source"]
            s_ev = [HAPropsSI("S", "H", h, "P", P_AMB, "T", T) for h, T in zip(h_ev, T_ev)]
            T_EvAvg = (h_ev[0] - h_ev[-1]) / (s_ev[0] - s_ev[-1])

            # Average sink temperature (Lorenz)
            h_cd, T_cd = cycle.air["h_sink"], cycle.air["T_sink"]
            s_cd = [HAPropsSI("S", "H", h, "P", P_AMB, "T", T) for h, T in zip(h_cd, T_cd)]
            T_CdAvg = (h_cd[-1] - h_cd[0]) / (s_cd[-1] - s_cd[0])

            # Lorenz COP and 2nd law efficiencies
            self.COP_Lorenz = T_CdAvg / (T_CdAvg - T_EvAvg)
            self.second_COP = self.COP / self.COP_Lorenz

            # Entropy consistency check
            self.COP_check = self.COP_Lorenz / (1 + (T_EvAvg * T_CdAvg) /
                               (T_CdAvg - T_EvAvg) * (self.sigma_total / Q_out))
            self.COP_error = abs(self.COP - self.COP_check) / self.COP

            # Volumetric flows at compressor inlets
            self.VdotUpper = cycle.high["m_dot"] / cycle.high["rho2"]
            self.VdotLower = cycle.low["m_dot"] / cycle.low["rho7"]
