# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:04:39 2025

@author: JHLam
"""

import time
from matplotlib import rc, rcParams
import HutamakiDryingIntegrated as Hutamaki
import SimpleCascade as HeatPump

# Plot settings
fs = 18
rcParams.update({'font.size': fs})
rc('xtick', labelsize=fs)
rc('ytick', labelsize=fs)
rcParams['figure.figsize'] = [10, 6]
rcParams['mathtext.fontset'] = 'stix'

start_time = time.time()         

# Calculate the drying model
Dryer = Hutamaki.HutamakiDryer(
    m_evap=0.15,                # kg/s
    m_recycled=5,               # kg/s
    T_make_up_air=15,           # °C
    R_make_up_air=0.6,          # Fraction
    T_heat_recovery_out=60,     # °C
    T_dew_point=85,             # °C
    T_dryer_out=120,            # °C
    Loss_dryer=0.2,             # Heat input lost in dryer as fraction of the total heat input
    p=101325,                   # Pa
    evaporator="mass water",    # Choose if evaporator stream is given in  'mass humid air' or 'mass water' or 'mass dry air' 
    recycled="mass humid air",  # Choose if recycled stream is given in  'mass humid air' of 'mass water' or 'mass dry air'
    plot_molier=True,
    plot_TQ=False,
)

print(f"The mass of humid air being recycled is: {Dryer.m_da6*(1+Dryer.pt6.W):.3f} kg/s")
print(f"omega_1 = {Dryer.pt1.W:.3f} kg/kg, omega_2 = {Dryer.pt2.W:.3f} kg/kg, omega_3 = {Dryer.pt3.W:.3f} kg/kg")
print(f"omega_4 = {Dryer.pt4.W:.3f} kg/kg, omega_5 = {Dryer.pt5.W:.3f} kg/kg, omega_6 = {Dryer.pt6.W:.3f} kg/kg")
print(f" enthalpy at point 3 = {Dryer.pt3.H/(1+Dryer.pt3.W)/1e3} and point 6 = {Dryer.pt6.H/(1+Dryer.pt6.W)/1e3}")
# print(f'The mass flow of dry air at point 1 is {Dryer.m_da1:.4f} kg/s,absolute humidity of {Dryer.pt1.W:.5f} \
# and temperature of {Dryer.Tc1:.2f} C. The air temperature out of the condenser is {Dryer.Tc2:.2f} C. The mass \
# flow of dry air into the evaporator is {Dryer.m_da4:.4f} kg/s,absolute humidity of {Dryer.pt4.W:.5f} and temperature \
# of {Dryer.Tc3:.2f} C.')


# Heat pump setup cascade pair 1
# mixture_low = ["Butane", "Hexane"]
# mole_fractions_low = [0.74, 0.26]
# mixture_high = ["DME", "Isopentane"]
# mole_fractions_high = [0.39, 0.61]
# T_start = 99.3

# # Heat pump setup cascade pair 2
mixture_low = ["Dimethyl Ether", "IsoPentane"]
mole_fractions_low = [0.62, 0.38]
mixture_high = ["IsoPentane", "Dimethyl Ether"]
mole_fractions_high = [0.55, 0.45]
T_start = 97.36

T_PinchInternal = 5  # K
T_PinchAir = 10  # K
etac = 0.80   # Isentropic efficiency

# Run the cascade heat pump optimization
print("Starting optimization...")
heatpump = HeatPump.SimpleCascade(
    mixture_low, mole_fractions_low,
    mixture_high, mole_fractions_high,
    T_PinchInternal,T_PinchAir, Dryer, 
    etac, T_start, IHX_upper=True,
    IHX_lower=True, plot=True, 
    PrintData=True, HTF=True, 
    ExtendedCalculations=True
)

# Show results
print(f"The second law efficency calculated throug COP and COP_Lorenz is {heatpump.second_COP*100:.2f}%")
print(f"The second law efficency calculated throug Exergy destruction and work is {heatpump.second_Ex*100:.2f}%")
print(f"The COP calculated throug entropy production is {heatpump.COP_check:.3f}")
print(f"The realative error between COP and COP check is {heatpump.COP_error*100:.3f}%")
print(f"\nOptimization complete in {time.time() - start_time:.2f} s")
print(f"The total power input in the upper cycle is {heatpump.W_upper/1000:.0f} and in the lower cycle {heatpump.W_lower/1000:.0f}")

print('\nUpper cycle:')
print(f'The pressure compressor in is {heatpump.cycle.high["P2"]/1e5:.3f} bar, the \
temperature compressor in is {heatpump.cycle.high["T2"]:.2f} K, the mass flow is \
{heatpump.cycle.high["m_dot"]:.3f} kg/s, the density compressor in is {heatpump.cycle.high["rho2"]:.4f} \
kg/m^3 and the volumetric flowrate compressor in is {heatpump.cycle.high["m_dot"]/heatpump.cycle.high["rho2"]:.3f} m^3/s')

print('\nLower cycle:')
print(f'The pressure compressor in is {heatpump.cycle.low["P7"]/1e5:.3f} bar, the \
temperature compressor in is {heatpump.cycle.low["T7"]:.2f} K, the mass flow is \
{heatpump.cycle.low["m_dot"]:.3f} kg/s, the density compressor in is {heatpump.cycle.low["rho7"]:.4f} \
kg/m^3 and the volumetric flowrate compressor in is {heatpump.cycle.low["m_dot"]/heatpump.cycle.low["rho7"]:.3f} m^3/s')

# Show results
print(f"\nData for Condenser calculation: \nP_h = {heatpump.cycle.high["P3"]/1e5:.4f} bar, T_h,in = {heatpump.cycle.high["T3"]:.4f} K, \
T_h,out = {heatpump.cycle.high["T4"]:.4f} K, m_dot_h = {heatpump.cycle.high["m_dot"]:.3f} kg/s, \
T_c,in = {heatpump.cycle.htf["T_sink"][0]:.2f} K, T_c,out = {heatpump.cycle.htf["T_sink"][-1]:.2f} K, \
m_dot_c = {heatpump.cycle.htf["m_dot_sink"]:.3f} kg/s")

print(f"\nData for upper IHX calculation: \nT_c,in = {heatpump.cycle.high["T1"]:.2f} K, \
T_c,out = {heatpump.cycle.high["T2"]:.2f} K, m_dot_c = {heatpump.cycle.high["m_dot"]:.3f} kg/s, \
T_h,in = {heatpump.cycle.high["T4"]:.2f} K, T_h,out = {heatpump.cycle.high["T5"]:.2f} K, \
m_dot_c = {heatpump.cycle.high["m_dot"]:.3f} kg/s")

print(f"\nData for shared HEX calculation: \nT_h,in = {heatpump.cycle.low["T8"]:.2f} K, \
T_h,out = {heatpump.cycle.low["T9"]:.2f} K, m_dot_h = {heatpump.cycle.low["m_dot"]:.3f} kg/s, \
T_c,in = {heatpump.cycle.high["T6"]:.2f} K, T_c,out = {heatpump.cycle.high["T1"]:.2f} K, \
m_dot_c = {heatpump.cycle.high["m_dot"]:.3f} kg/s")

print(f"\nData for lower IHX calculation: \nT_c,in = {heatpump.cycle.low["T6"]:.2f} K, \
T_c,out = {heatpump.cycle.low["T7"]:.2f} K, m_dot_c = {heatpump.cycle.low["m_dot"]:.3f} kg/s, \
T_h,in = {heatpump.cycle.low["T9"]:.2f} K, T_h,out = {heatpump.cycle.low["T10"]:.2f} K, \
m_dot_c = {heatpump.cycle.low["m_dot"]:.3f} kg/s") 

print(f"\nData for Evaporator calculation: \n  T_c,in = {heatpump.cycle.low["T11"]:.2f} K, \
T_c,out = {heatpump.cycle.low["T6"]:.2f} K, m_dot_c = {heatpump.cycle.low["m_dot"]:.3f} kg/s, \
T_h,in = {heatpump.cycle.htf["T_source"][-1]:.2f} K, T_h,out = {heatpump.cycle.htf["T_source"][0]:.2f} K, \
m_dot_h = {heatpump.cycle.htf["m_dot_source"]:.3f} kg/s")


