"""
Compressor.py
=============
Centrifugal compressor stage model based on one-dimensional loss correlations.

Each ``Compressor`` instance represents a single compressor stage and computes:
- Velocity triangles (inlet tip/hub, impeller exit)
- Impeller and diffuser geometry
- Seven internal loss contributions (disk friction, tip clearance, skin friction,
  blade loading, recirculation, mixing, and diffuser losses)
- Stage isentropic efficiency via an internal iteration loop

References
----------
Loss correlations follow the methodology described in:
    Aungier, R. H. (2000). *Centrifugal Compressors: A Strategy for Aerodynamic
    Design and Analysis*. ASME Press.

Dependencies
------------
- numpy
- CoolProp (with REFPROP backend)

Author
------
J.H. Lam
"""

import numpy as np
from CoolProp import PT_INPUTS, HmassP_INPUTS, PSmass_INPUTS


class Compressor:
    """
    One-dimensional model of a single centrifugal compressor stage.

    The constructor evaluates the full stage thermodynamics and geometry.
    Loss contributions and the resulting stage efficiency are stored as
    instance attributes.

    Parameters
    ----------
    fluid : CoolProp.AbstractState
        Configured CoolProp state object for the working fluid.
    m_dot : float
        Mass flow rate [kg/s].
    p_in : float
        Stage inlet pressure [Pa].
    T_in : float
        Stage inlet temperature [K].
    p_out : float
        Stage outlet pressure [Pa].
    eta_is : float
        Initial guess for the stage isentropic efficiency [-].
    N_s : float
        Specific speed used to size the impeller rotational speed in the
        first stage [-].
    first_stage : bool
        If True, ``omega`` is computed from ``N_s``; otherwise the supplied
        ``omega`` value is reused (shaft speed is fixed across stages).
    omega : float
        Rotational speed [rad/s]. Only used when ``first_stage`` is False.
    phi_u2 : float
        Velocity ratio  C_u2 / U2  (tangential velocity at impeller exit) [-].
    phi_r2 : float
        Velocity ratio  C_r2 / U2  (meridional velocity at impeller exit) [-].
    Z : int
        Number of impeller blades [-].
    x : float
        Blade thickness [m].
    t : float
        Tip clearance gap [m].
    D_ratio_tip_to_2 : float
        Diameter ratio  D_1,tip / D_2 [-].
    D_ratio_hub_to_tip : float
        Diameter ratio  D_1,hub / D_1,tip [-].
    b_star : float
        Blockage factor at the diffuser inlet [-].
    epsilon : float
        Jet-to-wake momentum ratio used in the mixing loss model [-].
    ratio_diffuser : float
        Diffuser-to-impeller exit diameter ratio  D_3 / D_2 [-].

    Attributes
    ----------
    omega : float
        Rotational speed [rad/s].
    U2 : float
        Blade tip speed at impeller exit [m/s].
    D2, D1_tip, D1_hub, D3 : float
        Key diameters [m].
    U1_tip, U1_hub : float
        Blade speeds at inlet tip and hub [m/s].
    C_r1, C_r2, C_u2, C2, W1_tip, W1_hub, W2 : float
        Velocity components in the velocity triangle [m/s].
    b2 : float
        Blade passage height at impeller exit [m].
    rho1, rho2 : float
        Fluid density at stage inlet and impeller exit [kg/m³].
    h_df, h_tc, h_sf, h_bl, h_rec, h_mix, h_diff : float
        Individual loss enthalpies [J/kg].
    h_loss : float
        Total loss enthalpy (sum of all contributions) [J/kg].
    eta_tt : float
        Total-to-total efficiency from the iterative loss loop [-].
    eta_calculated : float
        Stage isentropic efficiency inferred from the loss model [-].
    h_out : float
        Specific enthalpy at stage outlet [J/kg].
    T3 : float
        Outlet temperature (from initial isentropic guess) [K].
    """

    # ------------------------------------------------------------------
    # Construction / thermodynamic setup
    # ------------------------------------------------------------------

    def __init__(
        self,
        fluid,
        m_dot,
        p_in,
        T_in,
        p_out,
        eta_is,
        N_s,
        first_stage,
        omega,
        phi_u2,
        phi_r2,
        Z,
        x,
        t,
        D_ratio_tip_to_2,
        D_ratio_hub_to_tip,
        b_star,
        epsilon,
        ratio_diffuser,
    ):
        # ---- Inlet thermodynamic state ----
        fluid.update(PT_INPUTS, p_in, T_in)
        h1   = fluid.hmass()
        s1   = fluid.smass()
        rho1 = fluid.rhomass()
        V_dot = m_dot / rho1  # volumetric flow rate [m³/s]

        # ---- Isentropic outlet state ----
        fluid.update(PSmass_INPUTS, p_out, s1)
        h3s = fluid.hmass()

        # ---- Real outlet state (first pass, using eta_is guess) ----
        h3 = (h3s - h1) / eta_is + h1
        fluid.update(HmassP_INPUTS, h3, p_out)
        rho3      = fluid.rhomass()
        mu3       = fluid.viscosity()
        self.T3   = fluid.T()

        delta_h_is = h3s - h1   # isentropic specific work [J/kg]
        delta_h    = h3  - h1   # actual specific work      [J/kg]

        # ---- Rotational speed ----
        if first_stage:
            # Compute omega from specific speed correlation
            self.omega = np.ceil((N_s * delta_h_is**0.75) / V_dot**0.5)
        else:
            self.omega = omega

        # ---- Impeller tip speed and key diameters ----
        self.U2     = (delta_h / phi_u2) ** 0.5
        self.D2     = 2.0 * self.U2 / self.omega
        self.D1_tip = D_ratio_tip_to_2 * self.D2
        self.D1_hub = D_ratio_hub_to_tip * self.D1_tip
        self.D3     = ratio_diffuser * self.D2

        # ---- Velocity triangle ----
        self.U1_tip = self.omega * self.D1_tip / 2.0
        self.U1_hub = self.omega * self.D1_hub / 2.0

        # Meridional velocity at inlet (accounting for blade blockage)
        annulus_area   = (self.D1_tip**2 - self.D1_hub**2) * np.pi / 4.0
        blade_blockage = (self.D1_tip - self.D1_hub) / 2.0 * x * Z / 2.0
        self.C_r1 = m_dot / (rho1 * (annulus_area - blade_blockage))

        self.W1_tip = np.hypot(self.C_r1, -self.U1_tip)
        self.W1_hub = np.hypot(self.C_r1, -self.U1_hub)

        # Velocities at impeller exit
        self.C_r2 = phi_r2 * self.U2
        self.C_u2 = phi_u2 * self.U2
        self.C2   = np.hypot(self.C_r2, self.C_u2)
        self.W2   = np.hypot(self.C_r2, self.C_u2 - self.U2)

        self.rho1 = rho1

        # ---- Iterative loss / efficiency loop ----
        # The impeller-exit density (rho2) depends on the local efficiency,
        # which itself depends on the losses.  Ten iterations are sufficient.
        eta_tt = eta_is
        for _ in range(10):
            self.rho2, self.mu2 = self._find_rho2(fluid, p_in, T_in, delta_h, eta_tt)

            # Blade passage height at impeller exit
            self.b2 = m_dot / (self.rho2 * self.C_r2 * (np.pi * self.D2 - Z * x))

            # Evaluate all loss terms
            self.h_df   = self._disk_friction_loss(m_dot, rho1, self.rho2, self.mu2)
            self.h_tc   = self._tip_clearance_loss(rho1, self.rho2, t, Z)
            self.h_sf   = self._skin_friction_loss(Z)
            self.h_bl   = self._blade_loading_loss(Z)
            self.h_rec  = self._recirculation_loss(Z)
            self.h_mix  = self._mixing_loss(Z, epsilon, b_star)
            self.h_diff = self._diffuser_loss(
                m_dot, self.rho2, self.mu2, self.D2, self.b2,
                rho3, mu3, b_star, ratio_diffuser
            )

            # Update efficiency estimate (diffuser loss excluded from impeller eta)
            impeller_losses = (
                self.h_df + self.h_tc + self.h_sf
                + self.h_bl + self.h_rec + self.h_mix
            )
            eta_tt = (delta_h - impeller_losses) / delta_h

        # ---- Final performance metrics ----
        self.h_loss        = impeller_losses + self.h_diff
        self.eta_tt        = eta_tt
        self.eta_calculated = delta_h_is / (delta_h_is + self.h_loss)
        self.h_out         = (h3s - h1) / self.eta_calculated + h1

    # ------------------------------------------------------------------
    # Loss models
    # ------------------------------------------------------------------

    def _disk_friction_loss(self, m_dot, rho1, rho2, mu2):
        """
        Disk friction loss on the back face of the impeller.

        Parameters
        ----------
        m_dot : float
            Mass flow rate [kg/s].
        rho1, rho2 : float
            Fluid density at stage inlet and impeller exit [kg/m³].
        mu2 : float
            Dynamic viscosity at impeller exit [Pa·s].

        Returns
        -------
        float
            Specific enthalpy loss [J/kg].
        """
        Re = rho2 * self.U2 * self.D2 / (2.0 * mu2)
        f  = 0.0622 / Re**0.2
        return 0.5 * f * (rho1 + rho2) * self.D1_tip**2 * self.U2**3 / (16.0 * m_dot)

    def _tip_clearance_loss(self, rho1, rho2, t_clearance, N_blades):
        """
        Tip clearance leakage loss.

        Parameters
        ----------
        rho1, rho2 : float
            Fluid density at stage inlet and impeller exit [kg/m³].
        t_clearance : float
            Tip clearance gap [m].
        N_blades : int
            Number of impeller blades [-].

        Returns
        -------
        float
            Specific enthalpy loss [J/kg].
        """
        return (
            0.6
            * (t_clearance / self.b2)
            * self.C_u2
            * np.sqrt(
                2.0 * np.pi / (self.b2 * N_blades)
                * self.C_u2
                * self.C_r1
                * (self.D1_tip**2 - self.D1_hub**2)
                / ((self.D2 - self.D1_tip) * (1.0 + rho2 / rho1))
            )
        )

    def _skin_friction_loss(self, N_blades):
        """
        Blade passage skin friction loss.

        Parameters
        ----------
        N_blades : int
            Number of impeller blades [-].

        Returns
        -------
        float
            Specific enthalpy loss [J/kg].
        """
        C_f  = 0.006
        L_b  = self.D1_tip / 2.0 + self.D2 - self.D1_hub
        D_h  = (
            np.pi * (self.D1_tip**2 - self.D1_hub**2)
            / (np.pi * self.D1_tip + 2.0 * N_blades * (self.D1_tip - self.D1_hub))
        )
        W_bar = (2.0 * self.W2 + self.W1_tip + self.W1_hub) / 4.0
        return 2.0 * C_f * (L_b / D_h) * W_bar**2

    def _blade_loading_loss(self, N_blades):
        """
        Blade loading (diffusion factor) loss.

        Parameters
        ----------
        N_blades : int
            Number of impeller blades [-].

        Returns
        -------
        float
            Specific enthalpy loss [J/kg].
        """
        D_f = self._diffusion_factor(N_blades)
        return 0.05 * D_f**2 * self.U2**2

    def _recirculation_loss(self, N_blades):
        """
        Recirculation loss at the impeller inlet.

        Parameters
        ----------
        N_blades : int
            Number of impeller blades [-].

        Returns
        -------
        float
            Specific enthalpy loss [J/kg].
        """
        alpha = np.arctan(self.C_u2 / self.C_r2)
        D_f   = self._diffusion_factor(N_blades)
        return 8.0e-5 * np.sinh(3.5 * alpha**3) * D_f**2 * self.U2**2

    def _mixing_loss(self, N_blades, epsilon, b_star):
        """
        Jet-wake mixing loss at the impeller exit.

        Parameters
        ----------
        N_blades : int
            Number of impeller blades (unused in current formula, kept for
            interface consistency).
        epsilon : float
            Jet-to-wake momentum ratio [-].
        b_star : float
            Blockage factor at diffuser inlet [-].

        Returns
        -------
        float
            Specific enthalpy loss [J/kg].
        """
        alpha = np.arctan(self.C_u2 / self.C_r2)
        return (
            1.0 / (1.0 + np.tan(alpha)**2)
            * (((1.0 - epsilon) - b_star) / (1.0 - epsilon))**2
            * self.C2**2 / 2.0
        )

    def _diffuser_loss(
        self,
        m_dot,
        rho2,
        mu2,
        D2,
        b2,
        rho3,
        mu3,
        b_star,
        ratio_diffuser,
        b3=None,
        blockage3=0.0,
    ):
        """
        Vaneless diffuser friction loss.

        Computes the loss using a path-averaged meridional velocity with a
        mild compressibility correction and a turbulent friction factor.

        Parameters
        ----------
        m_dot : float
            Mass flow rate [kg/s].
        rho2, mu2 : float
            Density [kg/m³] and dynamic viscosity [Pa·s] at diffuser inlet.
        D2 : float
            Impeller exit diameter [m].
        b2 : float
            Blade passage height at impeller exit [m].
        rho3, mu3 : float
            Density [kg/m³] and dynamic viscosity [Pa·s] at diffuser exit.
        b_star : float
            Blockage factor at diffuser inlet [-].
        ratio_diffuser : float
            Diffuser-to-impeller exit diameter ratio  D_3 / D_2 [-].
        b3 : float or None, optional
            Blade passage height at diffuser exit [m].  Defaults to ``b_in``
            (parallel-wall assumption).
        blockage3 : float, optional
            Exit blockage fraction [-].  Default is 0 (no blockage).

        Returns
        -------
        float
            Specific enthalpy loss [J/kg].
        """
        r2  = 0.5 * D2
        r3  = ratio_diffuser * r2
        L   = max(r3 - r2, 1e-9)          # diffuser radial length [m]
        b_in = b_star * b2                 # effective inlet passage height [m]

        if b3 is None:
            b3 = b_in                      # parallel-wall default

        # ---- Exit area and velocities ----
        A3  = 2.0 * np.pi * r3 * b3 * (1.0 - blockage3)
        Cu3 = self.C_u2 * (r2 / r3)       # free-vortex swirl at exit
        Cr3 = m_dot / (rho3 * A3)         # continuity
        C3  = np.hypot(Cu3, Cr3)

        # ---- Path-mean meridional velocity ----
        # Incompressible: Cr(r) ≈ Cr2 · (r2/r)  →  average = Cr2·(r2/L)·ln(r3/r2)
        Cr_bar = (
            self.C_r2 * (r2 / L) * np.log(r3 / r2)
            * (rho2 / (0.5 * (rho2 + rho3)))   # density correction
        )

        # ---- Representative absolute velocity ----
        C_bar = 0.5 * (self.C2 + C3)

        # ---- Hydraulic diameter and mixture-mean fluid properties ----
        b_mean  = 0.5 * (b_in + b3)
        D_h     = 2.0 * b_mean
        rho_bar = 0.5 * (rho2 + rho3)
        mu_bar  = 0.5 * (mu2 + mu3)

        # ---- Friction factor (turbulent, clipped to physical range) ----
        Re = max(rho_bar * Cr_bar * D_h / max(mu_bar, 1e-12), 1e4)
        Cf = np.clip(0.015 * (1.8e5 / Re)**0.2, 5e-4, 8e-3)

        return 2.0 * Cf * (L / D_h) * C_bar**2

    # ------------------------------------------------------------------
    # Helper: diffusion factor (shared by blade-loading and recirculation)
    # ------------------------------------------------------------------

    def _diffusion_factor(self, N_blades):
        """
        Lieblein diffusion factor for the impeller.

        Parameters
        ----------
        N_blades : int
            Number of impeller blades [-].

        Returns
        -------
        float
            Diffusion factor D_f [-].
        """
        blade_term = (
            N_blades / np.pi * (1.0 - self.D1_tip / self.D2)
            + self.D1_tip / self.D2
        )
        return (
            1.0
            - self.W2 / self.W1_tip
            + (0.75 * self.C_u2 / self.U2)
            / (self.W1_tip / self.W2 * blade_term)
        )

    # ------------------------------------------------------------------
    # Helper: iterative outlet density calculation
    # ------------------------------------------------------------------

    def _find_rho2(self, fluid, p_in, T_in, delta_h, eta_tt):
        """
        Estimate the density and viscosity at the impeller exit.

        Uses an inner iteration to find the total outlet pressure consistent
        with the current total-to-total efficiency, then converts to static
        conditions via an isentropic relation.

        Parameters
        ----------
        fluid : CoolProp.AbstractState
            Configured CoolProp state object for the working fluid.
        p_in : float
            Stage inlet pressure [Pa].
        T_in : float
            Stage inlet temperature [K].
        delta_h : float
            Actual specific work input [J/kg].
        eta_tt : float
            Current total-to-total efficiency estimate [-].

        Returns
        -------
        rho2 : float
            Static density at impeller exit [kg/m³].
        mu2 : float
            Dynamic viscosity at impeller exit [Pa·s].
        """
        # ---- Inlet state ----
        fluid.update(PT_INPUTS, p_in, T_in)
        h1 = fluid.hmass()
        s1 = fluid.smass()

        # ---- Total enthalpies ----
        h0_1    = h1 + 0.5 * self.C_r1**2
        h0_2    = h0_1 + delta_h
        h0_2_is = h0_1 + eta_tt * delta_h

        # ---- Iterate for total outlet pressure ----
        p0_2 = p_in
        for _ in range(20):
            fluid.update(PSmass_INPUTS, p0_2, s1)
            h_is = fluid.hmass()
            if abs(h_is - h0_2_is) / h0_2_is < 1e-5:
                break
            p0_2 *= h0_2_is / h_is

        # ---- Convert total to static conditions ----
        fluid.update(HmassP_INPUTS, h0_2, p0_2)
        T0_2 = fluid.T()

        fluid.update(PT_INPUTS, p0_2, T0_2)
        cp2   = fluid.cpmass()
        gamma = cp2 / fluid.cvmass()
        T2    = T0_2 - self.C2**2 / (2.0 * cp2)
        p2    = p0_2 / (T0_2 / T2) ** (gamma / (gamma - 1.0))

        fluid.update(PT_INPUTS, p2, T2)
        return fluid.rhomass(), fluid.viscosity()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        lines = [f"{k} = {v:.6g}" if isinstance(v, float) else f"{k} = {v}"
                 for k, v in self.__dict__.items()]
        return "\n".join(lines)
