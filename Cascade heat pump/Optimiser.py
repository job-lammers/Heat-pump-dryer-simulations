# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:04:39 2025

@author: JHLam
"""

import time
from matplotlib import rc, rcParams
import DryingIntegrated
import SimpleCascade as HeatPump

# Plot settings
fs = 18
rcParams.update({"font.size": fs})
rc("xtick", labelsize=fs)
rc("ytick", labelsize=fs)
rcParams["figure.figsize"] = [10, 6]
rcParams["mathtext.fontset"] = "stix"

start_time = time.time()

# ---------------------------------------------------------
# Drying model
# ---------------------------------------------------------
Dryer = DryingIntegrated.Dryer(
    m_evap=0.15,                # kg/s
    m_recirculated=5,           # kg/s
    T_make_up_air=15,           # °C
    R_make_up_air=0.6,          # -
    T_heat_recovery_out=60,     # °C
    T_dew_point=85,             # °C
    T_dryer_out=120,            # °C
    Loss_dryer=0.2,             # -
    p=101325,                   # Pa
    evaporator="mass water",
    recirculated="mass humid air",
    plot_molier=True,
)

# ---------------------------------------------------------
# Heat pump setup
# ---------------------------------------------------------
mixture_low = ["Butane", "Hexane"]
mole_fractions_low = [0.74, 0.26]
mixture_high = ["DME", "Isopentane"]
mole_fractions_high = [0.39, 0.61]
T_start = 99.3

T_PinchInternal = 5  # K
T_PinchAir = 10      # K
etac = 0.80          # -

print("Starting optimization...")
heatpump = HeatPump.SimpleCascade(
    mixture_low,
    mole_fractions_low,
    mixture_high,
    mole_fractions_high,
    T_PinchInternal,
    T_PinchAir,
    Dryer,
    etac,
    T_start,
    IHX_upper=True,
    IHX_lower=True,
    plot=True,
    HTF=True,
    ExtendedCalculations=True,
)

# ---------------------------------------------------------
# Essential output
# ---------------------------------------------------------
print("\n--- Heat pump performance ---")
print(f"COP:                        {heatpump.COP:.3f}")
print(f"COP check (entropy method): {heatpump.COP_check:.3f}")
print(f"COP discrepancy:            {heatpump.COP_error*100:.3f} %")
print(f"Second-law efficiency:      {heatpump.second_COP*100:.2f} %")

# ---------------------------------------------------------
# Key cycle points 
# ---------------------------------------------------------
high = heatpump.cycle.high
low = heatpump.cycle.low


print("\n--- Upper cycle working fluid ---")
print(f"Evaporator out (1):     T = {high['T1']:.2f} K, p = {high['P1']/1e5:.3f} bar")
print(f"Compressor in (2):      T = {high['T2']:.2f} K, p = {high['P2']/1e5:.3f} bar")
print(f"Compressor out (3):     T = {high['T3']:.2f} K, p = {high['P3']/1e5:.3f} bar")
print(f"Condenser out (4):      T = {high['T4']:.2f} K, p = {high['P4']/1e5:.3f} bar")

if heatpump.IHX_upper:
    ls = 7    
    print(f"Expansion valve in (5): T = {high['T5']:.2f} K, p = {high['P5']/1e5:.3f} bar")
    print(f"Evaporator in (6):      T = {high['T6']:.2f} K, p = {high['P2']/1e5:.3f} bar")
else:
    ls = 6
    print(f"Expansion valve in (5): T = {high['T5']:.2f} K, p = {high['P2']/1e5:.3f} bar")

print(f"Mass flow upper cycle:  {high['m_dot']:.3f} kg/s")



print("\n--- Lower cycle working fluid ---")
print(f"Evaporator out ({ls}):         T = {low['T6']:.2f} K, p = {low['P6']/1e5:.3f} bar")
print(f"Compressor in ({ls+1}):          T = {low['T7']:.2f} K, p = {low['P7']/1e5:.3f} bar")
print(f"Compressor out ({ls+2}):         T = {low['T8']:.2f} K, p = {low['P8']/1e5:.3f} bar")
print(f"Condenser out ({ls+3}):         T = {low['T9']:.2f} K, p = {low['P9']/1e5:.3f} bar")

if heatpump.IHX_lower:
    print(f"Expansion valve in ({ls+4}):    T = {low['T10']:.2f} K, p = {low['P10']/1e5:.3f} bar")
    print(f"Evaporator in ({ls+5}):         T = {low['T11']:.2f} K, p = {low['P11']/1e5:.3f} bar")
else:
    print(f"Expansion in ({ls+4}):          T = {low['T10']:.2f} K, p = {low['P10']/1e5:.3f} bar")

print(f"Mass flow lower cycle:      {low['m_dot']:.3f} kg/s")

# ---------------------------------------------------------
# Volumetric flow and runtime
# ---------------------------------------------------------
print(f"\nVolumetric flow at upper compressor inlet: {heatpump.VdotUpper:.4f} m³/s")
print(f"Volumetric flow at lower compressor inlet: {heatpump.VdotLower:.4f} m³/s")

print(f"\nOptimisation executed in {time.time() - start_time:.2f} s")





