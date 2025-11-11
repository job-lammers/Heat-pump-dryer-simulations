
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 11:06:02 2025

@author: JHLam
"""
from CoolProp.CoolProp import HAPropsSI
from single_stage_model.constants import P_AMB
from CoolProp import AbstractState as AS
from CoolProp import PT_INPUTS
from single_stage_model.utils import EntropyProduction
from single_stage_model.core import HeatPumpCycle
from single_stage_model.cyclestate import CycleState
from single_stage_model.plots import (
    plot_Ts_diagram_with_dome,
    plot_TQ_diagram_all,
)


class SimpleSingleStage:
    def __init__(self, mixture, mole_fractions, T_PinchInternal, T_PinchAir,
                 dryer, etac, IHX, plot, HTF, ExtendedCalculations):
        self.mixture = mixture
        self.mole_fractions = mole_fractions
        self.dryer = dryer
        self.T_PinchInternal = T_PinchInternal
        self.T_PinchAir = T_PinchAir
        self.etac = etac
        self.IHX = IHX
        self.plot = plot
        self.HTF = HTF
        self.ExtendedCalculations = ExtendedCalculations

        self.optimize_cycle()

    def optimize_cycle(self):
        # Refrigerant mixture setup
        Refrigerant = AS("REFPROP", "&".join(self.mixture))
        Refrigerant.set_mole_fractions(self.mole_fractions)

        # Cycle container
        cycle = CycleState()

        # Run main heat pump model
        self.IHX = HeatPumpCycle(
            Refrigerant,
            cycle,
            self.dryer,
            self.T_PinchInternal,
            self.T_PinchAir,
            self.etac,
            self.IHX,
            self.HTF,
        )

        # Optional plots
        if self.plot:
            plot_Ts_diagram_with_dome(cycle, Refrigerant, self.IHX)
            # Source/sink T-Q
            plot_TQ_diagram_all(cycle, self.dryer, plt_sink=True, plt_source=True, IHX=False, HTF=self.HTF)
            # IHX only
            if self.IHX:
                plot_TQ_diagram_all(cycle, self.dryer, plt_sink=False, plt_source=False, IHX=True, HTF=False)

        # COP from condenser heat / compressor work
        Q_out = (cycle.ref["h3"] - cycle.ref["h4"]) * cycle.ref["m_dot"]
        self.W = (cycle.ref["h3"] - cycle.ref["h2"]) * cycle.ref["m_dot"]
        self.COP = Q_out / self.W

        # Entropy production of components
        self.sigma_total, self.entropies = EntropyProduction(
            cycle, self.dryer, self.IHX, self.HTF, print_entropy_production=True
        )

        # Some useful cycle data
        self.PR = cycle.ref["P3"] / cycle.ref["P2"]
        self.P_max = cycle.ref["P3"]

        # Extended thermodynamic indicators
        if self.ExtendedCalculations:
            # Average source temperature (Lorenz)
            h_ev = cycle.air["h_source"]
            T_ev = cycle.air["T_source"]
            s_ev = [HAPropsSI("S", "H", h, "P", P_AMB, "T", T) for h, T in zip(h_ev, T_ev)]
            T_EvAvg = (h_ev[0] - h_ev[-1]) / (s_ev[0] - s_ev[-1])

            # Average sink temperature (Lorenz)
            h_cd = cycle.air["h_sink"]
            T_cd = cycle.air["T_sink"]
            s_cd = [HAPropsSI("S", "H", h, "P", P_AMB, "T", T) for h, T in zip(h_cd, T_cd)]
            T_CdAvg = (h_cd[-1] - h_cd[0]) / (s_cd[-1] - s_cd[0])

            # Lorenz COP and 2nd law efficiencies
            self.COP_Lorenz = T_CdAvg / (T_CdAvg - T_EvAvg)
            self.second_COP = self.COP / self.COP_Lorenz
            self.second_Ex = 1 - (T_EvAvg * self.sigma_total) / self.W

            # Consistency check
            self.COP_check = self.COP_Lorenz / (
                1
                + (T_EvAvg * T_CdAvg) / (T_CdAvg - T_EvAvg) * (self.sigma_total / Q_out)
            )
            self.COP_error = abs(self.COP - self.COP_check) / self.COP

            # Volumetric flow at condenser inlet
            Refrigerant.update(PT_INPUTS, cycle.ref["P2"], cycle.ref["T2"])
            rho2 = Refrigerant.rhomass()
            self.Vdot = cycle.ref["m_dot"] / rho2

        # expose cycle
        self.cycle = cycle

