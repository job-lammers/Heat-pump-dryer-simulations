import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.legend_handler import HandlerPatch
import CoolProp.CoolProp as CP

# ----------------------------------------------------------------------------- 
# Constants
# -----------------------------------------------------------------------------
KELVIN = 273.15  # °C → K offset


# ----------------------------------------------------------------------------- 
# Custom legend handler so arrow objects render correctly in the legend
# -----------------------------------------------------------------------------
class HandlerArrow(HandlerPatch):
    """Legend handler to display FancyArrowPatch objects nicely."""
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width,
                       height, fontsize, trans):
        center = (xdescent + width / 2, ydescent + height / 2)
        arrow = FancyArrowPatch(
            (xdescent, center[1]),
            (xdescent + width, center[1]),
            arrowstyle=orig_handle.get_arrowstyle(),
            mutation_scale=fontsize,
            lw=orig_handle.get_linewidth(),
            color=orig_handle.get_edgecolor(),
        )
        return [arrow]

class ThermoStateHumid:
    """
    Humid-air state for the dryer.
    Supported pairs:
    - "TR"    : temperature + relative humidity
    - "TTdp"  : temperature + dew point
    - "HW"    : enthalpy + humidity ratio
    """
    def __init__(self, pair, value1, value2, pressure=101325):
        self.P = pressure

        if pair == "TR":
            # value1 = T [K], value2 = RH [-]
            self.T = value1
            self.R = value2
            self.H = CP.HAPropsSI("H", "T", self.T, "R", self.R, "P", self.P)
            self.W = CP.HAPropsSI("W", "T", self.T, "R", self.R, "P", self.P)
            self.Tdp = CP.HAPropsSI("Tdp", "T", self.T, "R", self.R, "P", self.P)
            self.S = CP.HAPropsSI("S", "T", self.T, "R", self.R, "P", self.P)

        elif pair == "TTdp":
            # value1 = T [K], value2 = dew point [K]
            self.T = value1
            self.Tdp = value2
            self.W = CP.HAPropsSI("W", "T", self.T, "Tdp", self.Tdp, "P", self.P)
            self.R = CP.HAPropsSI("R", "T", self.T, "Tdp", self.Tdp, "P", self.P)
            self.H = CP.HAPropsSI("H", "T", self.T, "Tdp", self.Tdp, "P", self.P)
            self.S = CP.HAPropsSI("S", "T", self.T, "Tdp", self.Tdp, "P", self.P)
            
        elif pair == "HW":
            # value1 = H [J/kg dry air], value2 = W [kg/kg]
            self.H = value1
            self.W = value2
            self.R = CP.HAPropsSI("R", "W", self.W, "H", self.H, "P", self.P)
            self.T = CP.HAPropsSI("T", "W", self.W, "H", self.H, "P", self.P)
            self.Tdp = CP.HAPropsSI("Tdp", "W", self.W, "H", self.H, "P", self.P)
            self.S = CP.HAPropsSI("S", "W", self.W, "H", self.H, "P", self.P)
            
        else:
            raise ValueError(f"ThermoStateHumid: pair '{pair}' not implemented for dryer.")

class Dryer:
    """
    Represents the drying process and can plot the Mollier diagram.
    """

    def __init__(self, m_evap, m_recirculated, T_make_up_air, R_make_up_air, T_heat_recovery_out,
                 T_dew_point, T_dryer_out, Loss_dryer, p, evaporator, recirculated, plot_molier):
        # Store inputs
        self.m_evap = m_evap
        self.m_recirculated = m_recirculated
        self.T_make_up_air = T_make_up_air
        self.R_make_up_air = R_make_up_air
        self.T_heat_recovery_out = T_heat_recovery_out
        self.T_dew_point = T_dew_point
        self.T_dryer_out = T_dryer_out
        self.Loss_dryer = Loss_dryer
        self.p = p
        self.evaporator = evaporator
        self.recirculated = recirculated

        # Compute all state points for the drying cycle
        self.calc_points_drying(
            m_evap, m_recirculated, T_make_up_air, R_make_up_air,
            T_heat_recovery_out, T_dew_point, T_dryer_out,
            Loss_dryer, p, evaporator, recirculated
        )

        if plot_molier:
            self.plot_molier()

    def calc_points_drying(self, m_evap, m_recirculated, T_make_up_air, R_make_up_air,
                           T_heat_recovery_out, T_dew_point, T_dryer_out,
                           Loss_dryer, p, evaporator, recirculated):
        """Calculate all psychrometric points for the drying loop."""

        # 3: stream out of the dryer
        self.pt3 = ThermoStateHumid("TTdp", T_dryer_out + KELVIN, T_dew_point + KELVIN)
        self.Tc3 = self.pt3.T - KELVIN

        # Flow into the evaporator
        if evaporator == "mass humid air":
            self.m_da4 = m_evap / (self.pt3.W + 1)
        elif evaporator == "mass water":
            self.m_da4 = m_evap / self.pt3.W
        elif evaporator == "mass dry air":
            self.m_da4 = m_evap
        else:
            self.m_da4 = m_evap

        # 4: saturated at pt3 dew point
        self.pt4 = ThermoStateHumid("TR", self.pt3.Tdp, 1)
        self.Tc4 = self.pt4.T - KELVIN

        # Mass balance at dryer outlet
        if recirculated == "mass humid air":
            self.m_da3 = self.m_da4 + m_recirculated / (self.pt3.W + 1)
        elif recirculated == "mass water":
            self.m_da3 = self.m_da4 + m_recirculated / self.pt3.W
        elif recirculated == "mass dry air":
            self.m_da3 = self.m_da4 + m_recirculated
        else:
            self.m_da3 = self.m_da4 + m_recirculated

        # 6: make-up air
        self.pt6 = ThermoStateHumid("TR", T_make_up_air + KELVIN, R_make_up_air)
        self.Tc6 = self.pt6.T - KELVIN
        self.m_da6 = self.m_da4  # compensate for dry air leaving

        # 1: mixed stream into condenser
        self.m_da1 = self.m_da3
        self.m_da_recycle = self.m_da3 - self.m_da4
        self.pt1 = ThermoStateHumid(
            "HW",
            (self.pt3.H * self.m_da_recycle + self.pt6.H * self.m_da6) / (self.m_da_recycle + self.m_da6),
            (self.m_da_recycle * self.pt3.W + self.m_da6 * self.pt6.W) / (self.m_da_recycle + self.m_da6),
        )
        self.Tc1 = self.pt1.T - KELVIN

        # 2: after condenser
        self.m_da2 = self.m_da3
        Q_heat = (self.pt3.H - self.pt1.H) * (self.m_da2 / (1 - Loss_dryer))
        self.pt2 = ThermoStateHumid("HW", self.pt1.H + Q_heat / self.m_da2, self.pt1.W)
        self.Tc2 = self.pt2.T - KELVIN

        # 5: out of evaporator / heat recovery
        self.pt5 = ThermoStateHumid("TR", T_heat_recovery_out + KELVIN, 1)
        self.Tc5 = self.pt5.T - KELVIN

        # sensible heat recovered
        # self.Q_super_heat = (self.pt4.H - self.pt5.H) * self.m_da1
        
        self.Q_heating = (self.pt2.H - self.pt1.H) * self.m_da1

    def plot_molier(self):
        """Plot the drying process on a Mollier diagram (h–ω style)."""
        fig, ax = plt.subplots(figsize=(10 * 0.8, 7 * 0.8))

        Tdbvec = np.linspace(-10, 90) + KELVIN
        Tdb_ticks = np.arange(0, self.pt2.T - KELVIN + 50, 25)
        omega_ticks = np.arange(0, 1.1, 0.1)

        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()

        omega_sat = CP.HAPropsSI("W", "T", Tdbvec, "P", self.p, "R", 1)
        ax.plot(omega_sat, Tdbvec - KELVIN, "k", lw=1.5)

        # isenthalpic lines (only for points 1 and 2)
        h_list = [self.pt1.H / 1e3, self.pt2.H / 1e3]
        for h in h_list:
            R = np.linspace(1, 0, 50)
            if h < 400:
                Tdb = CP.HAPropsSI("Tdb", "H", h * 1e3, "P", self.p, "R", R)
                omega = CP.HAPropsSI("W", "Tdb", Tdb, "P", self.p, "R", R)
                ax.plot(omega, Tdb - KELVIN, "b", lw=1, ls="--", label="Isenthalpic line [kJ/kg]")
            else:
                T_db_start = CP.HAPropsSI("Tdp", "R", 1, "P", self.p, "H", h * 1e3)
                Tdb = np.linspace(T_db_start, Tdb_ticks[-1] + KELVIN, 50)
                omega = CP.HAPropsSI("W", "Tdb", Tdb, "P", self.p, "H", h * 1e3)
                ax.plot(
                    omega, Tdb - KELVIN, "b", lw=1, ls="--",
                    label="Isenthalpic lines [kJ/kg dry air]" if h == h_list[0] else None
                )

            dx = omega[0] - omega[1]
            dy = (Tdb[0] - KELVIN) - (Tdb[1] - KELVIN)
            angle = np.degrees(np.arctan2(dy, dx))
            if h >= h_list[0]:
                ax.text(
                    omega[1] + 0.007,
                    (Tdb[1] - KELVIN) - 7,
                    f"{np.round(h, 1)}",
                    fontsize=16,
                    color="b",
                    verticalalignment="center",
                    rotation=angle,
                    rotation_mode="anchor",
                    transform_rotates_text=True,
                )

        # horizontal helper lines for points 1–3
        omega_end1 = CP.HAPropsSI("W", "T", self.pt1.T, "P", self.p, "H", self.pt1.H)
        ax.hlines(self.Tc1, 0, omega_end1, colors="red", linestyles="dashed", linewidth=1)
        ax.text(0.3, self.Tc1 - 1, r"$ϑ_{db}$ = " f"{np.round(self.Tc1, 1)}" r"$^{\circ}C$", fontsize=16, color="red",
                verticalalignment="top", horizontalalignment="left")

        omega_end2 = CP.HAPropsSI("W", "T", self.pt2.T, "P", self.p, "H", self.pt2.H)
        ax.hlines(self.Tc2, 0, omega_end2, colors="red", linestyles="dashed", linewidth=1)
        ax.text(0.3, self.Tc2, r"$ϑ_{db}$ = " f"{np.round(self.Tc2, 1)}" r"$^{\circ}C$", fontsize=16, color="red",
                verticalalignment="bottom", horizontalalignment="left")

        omega_end3 = CP.HAPropsSI("W", "T", self.pt3.T, "P", self.p, "H", self.pt3.H)
        ax.hlines(self.Tc3, 0, omega_end3, colors="red", linestyles="dashed", linewidth=1)
        ax.text(0.3, self.Tc3 + 1, r"$ϑ_{db}$ = " f"{np.round(self.Tc3, 1)}" r"$^{\circ}C$", fontsize=16, color="red",
                verticalalignment="bottom", horizontalalignment="left")

        # gridlines
        for Tdb_C in Tdb_ticks:
            if Tdb_C <= 90:
                omega_end = CP.HAPropsSI("W", "T", Tdb_C + KELVIN, "P", self.p, "R", 1)
            else:
                omega_end = omega_ticks[-1]
            ax.hlines(Tdb_C, 0, omega_end, colors="gray", linestyles="dashed", linewidth=0.5)

        h_grid_list = [0, 400, 700, 1000, 1200, 1600, 2000, 2400, 2800, 3000, 3200]
        for i, omega_val in enumerate(omega_ticks):
            omega_start = CP.HAPropsSI("W", "T", KELVIN, "P", self.p, "R", 1)
            if omega_val < omega_start:
                Tdb_start = 0
            else:
                Tdb_start = CP.HAPropsSI("Tdp", "W", omega_val, "P", self.p, "H", h_grid_list[i] * 1e3) - KELVIN
            ax.vlines(omega_val, Tdb_start, Tdb_ticks[-1], colors="gray", linestyles="dashed", linewidth=0.5)

        # points
        ax.scatter(
            [self.pt1.W, self.pt2.W, self.pt3.W, self.pt4.W, self.pt5.W, self.pt6.W],
            [self.Tc1, self.Tc2, self.Tc3, self.Tc4, self.Tc5, self.Tc6],
            color="black", zorder=5
        )
        ax.text(self.pt1.W + 0.005, self.Tc1 - 4, "1", fontsize=16)
        ax.text(self.pt2.W + 0.01, self.Tc2 - 5, "2", fontsize=16)
        ax.text(self.pt3.W + 0.01, self.Tc3 + 1, "3", fontsize=16)
        ax.text(self.pt4.W + 0.01, self.Tc4 + 1, "4", fontsize=16)
        ax.text(self.pt5.W, self.Tc5 - 4, "5", fontsize=16)
        ax.text(self.pt6.W + 0.01, self.Tc6 - 2, "6", fontsize=16)

        # 100% RH curve between 4 and 5
        Tdb_recovery = [self.Tc3, self.Tc4]
        Tdb_curve = np.linspace(self.Tc4, self.Tc5)
        Tdb_recovery.extend(Tdb_curve)
        W_recovery = [self.pt3.W, self.pt4.W]
        W_curve = CP.HAPropsSI("W", "Tdb", Tdb_curve + KELVIN, "R", 1, "P", self.p)
        W_recovery.extend(W_curve)
        ax.plot(W_recovery, Tdb_recovery, "b-", lw=3)

        # arrows
        ax.annotate("", xy=(self.pt3.W, self.Tc3), xytext=(self.pt2.W, self.Tc2),
                    arrowprops=dict(arrowstyle="->", color="green", lw=3, ls="-", mutation_scale=20))
        ax.annotate("", xy=(self.pt2.W, self.Tc2), xytext=(self.pt1.W, self.Tc1),
                    arrowprops=dict(arrowstyle="->", color="red", lw=3, mutation_scale=20))
        ax.annotate("", xy=(self.pt1.W, self.Tc1), xytext=(self.pt3.W, self.Tc3),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2, ls="--", mutation_scale=20))
        ax.annotate("", xy=(self.pt1.W, self.Tc1), xytext=(self.pt6.W, self.Tc6),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2, ls="--", mutation_scale=20))
        ax.annotate("", xy=(W_curve[-1], self.Tc5), xytext=(W_curve[-5], Tdb_curve[-5]),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=3, mutation_scale=20))

        # legend
        arrow1 = FancyArrowPatch((0, 0), (1, 0), arrowstyle="->", color="red", lw=3)
        arrow2 = FancyArrowPatch((0, 0), (1, 0), arrowstyle="->", color="green", lw=3)
        arrow3 = FancyArrowPatch((0, 0), (1, 0), arrowstyle="->", color="blue", lw=3)
        handles = [arrow1, arrow2, arrow3]
        labels = ["Air heated in condenser", "Non-isenthalpic drying", "Heat recovered in evaporator"]

        ax.legend(
            handles + ax.get_legend_handles_labels()[0],
            labels + ax.get_legend_handles_labels()[1],
            handler_map={FancyArrowPatch: HandlerArrow()},
            loc="upper left", bbox_to_anchor=(1, 1), fontsize=15
        )

        # axes
        ax.set_xlabel(r"Absolute Humidity ($\omega_{hum}$) [kg/kg]")
        ax.set_ylabel(r"Dry Bulb Temperature ($ϑ_{db}$) [$^{\circ}$C]")
        ax.set_xlim(omega_ticks[0], omega_ticks[-1])
        ax.set_ylim(Tdb_ticks[0], Tdb_ticks[-1])
        ax.set_yticks(Tdb_ticks)
        ax.set_xticks(omega_ticks)

        plt.show()



    








