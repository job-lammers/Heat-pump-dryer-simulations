# -*- coding: utf-8 -*-
"""
@author: JHLam
"""
import time
from matplotlib import rc, rcParams
import DryingIntegrated
import SimpleSingleStage as HeatPump

# Plot settings
fs = 16
rcParams.update({'font.size': fs})
rc('xtick', labelsize=fs)
rc('ytick', labelsize=fs)
rcParams['figure.figsize'] = [10, 6]
rcParams['mathtext.fontset'] = 'stix'

start_time = time.time()

# -------------------------------------------------------------------------
# Drying model
# -------------------------------------------------------------------------
Dryer = DryingIntegrated.Dryer(
    m_evap=0.15,                 # kg/s
    m_recirculated=5,            # kg/s
    T_make_up_air=15,            # °C
    R_make_up_air=0.6,           # -
    T_heat_recovery_out=60,      # °C
    T_dew_point=85,              # °C
    T_dryer_out=120,             # °C
    Loss_dryer=0.2,              # -
    p=101325,                    # Pa
    evaporator="mass water",
    recirculated="mass humid air",
    plot_molier=True,
)

# -------------------------------------------------------------------------
# Heat pump setup
# -------------------------------------------------------------------------
mixture = ["Butane", "Hexane"]
mole_fractions = [0.51, 0.49]
T_PinchInternal = 5   # K
T_PinchAir = 10       # K
etac = 0.80           # isentropic efficiency

print("Starting heat pump calculation...")
heatpump = HeatPump.SimpleSingleStage(
    mixture,
    mole_fractions,
    T_PinchInternal,
    T_PinchAir,
    Dryer,
    etac,
    IHX=True,
    plot=True,
    HTF=True,
    ExtendedCalculations=True,
)

# -------------------------------------------------------------------------
# Essential output only
# -------------------------------------------------------------------------
# Main performance
print("\n--- Heat pump performance ---")
print(f"COP:                        {heatpump.COP:.3f}")
print(f"COP check (entropy method): {heatpump.COP_check:.3f}")
print(f"COP discrepancy:            {heatpump.COP_error*100:.3f} %")
print(f"Second-law efficiency:      {heatpump.second_COP*100:.2f} %")


# Minimal cycle sanity check
ref = heatpump.cycle.ref
htf = heatpump.cycle.htf

if heatpump.IHX == True:
    print("\n--- Key cycle points of working fluid ---")
    print(f"Evaporator out (1):     T = {ref['T1']:.2f} K, p = {ref['P1']/1e5:.3f} bar")
    print(f"Compressor in (2):      T = {ref['T2']:.2f} K, p = {ref['P2']/1e5:.3f} bar")
    print(f"Compressor out (3):     T = {ref['T3']:.2f} K, p = {ref['P3']/1e5:.3f} bar")
    print(f"Condenser out (4):      T = {ref['T4']:.2f} K, p = {ref['P4']/1e5:.3f} bar")
    print(f"Expansion valve in (5): T = {ref['T5']:.2f} K, p = {ref['P5']/1e5:.3f} bar")
    print(f"Evaporator in (6):      T = {ref['T6']:.2f} K, p = {ref['P6']/1e5:.3f} bar")
    print(f"Mass flow:              {ref['m_dot']:.3f} kg/s")
else:
    print("\n--- Key cycle points ---")
    print(f"Evaporator out (1):     T = {ref['T1']:.2f} K, p = {ref['P1']/1e5:.3f} bar")
    print(f"Compressor in (2):      T = {ref['T2']:.2f} K, p = {ref['P2']/1e5:.3f} bar")
    print(f"Compressor out (3):     T = {ref['T3']:.2f} K, p = {ref['P3']/1e5:.3f} bar")
    print(f"Condenser out (4):      T = {ref['T4']:.2f} K, p = {ref['P4']/1e5:.3f} bar")
    print(f"Evaporator in (5):      T = {ref['T5']:.2f} K, p = {ref['P5']/1e5:.3f} bar")
    print(f"Mass flow:              {ref['m_dot']:.3f} kg/s")
    
# Volumentric flow at compressor inlet
print(f"\nVolumetric flow at compressor inlet: {heatpump.Vdot:.4f} m³/s")

# Runtime
print(f"\nOptimisation executed in {time.time() - start_time:.2f} s")
