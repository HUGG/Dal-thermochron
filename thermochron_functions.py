#!/usr/bin/env python3
"""Functions for thermochronometer age prediction."""

# Load modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.interpolate import interp1d

# Exhumation rate from gradient and age
def edot_linear(ages=[10.0], closure=100.0, gradient=10.0):
    edots = []
    for age in ages:
        edots.append((closure/gradient)/age)
    return edots

def temp_1D(z, time, g_initial, vz, kappa):
    """Calculate solution to 1D transient advection-diffusion heat transfer equation.
    
    Keyword arguments:
    z -- array of depths from surface (units: km)
    time -- array of time since model start (units: m)
    g_initial -- initial temperature gradient (units: deg. C / km)
    vz -- vertical advection velocity (units: km / Ma)
    kappa -- thermal diffusivity (units: km^2 / Ma)
    """
    # Calculate T separately for case where t = 0 to avoid divide-by-zero warnings
    if time == 0:
        temperature = g_initial * z
    else:
        temperature = g_initial * (z + vz * time) + (g_initial / 2.0) * ((z - vz * time) *\
                      np.exp(-(vz * z) / kappa) * erfc((z - vz * time) / (2.0 * np.sqrt(kappa *time))) -\
                      (z + vz * time) * erfc((z + vz * time) / (2.0 * np.sqrt(kappa * time))))
    return temperature

def dodson(tau, temp_hist, temp_hist_prev, time, time_prev, age, tc_prev, ea, dtdt, a, d0a2, r):
    """Calculate thermochronometer age and closure temperature using Dodson's method.
    
    Keyword arguments:
    tau -- diffusivity characteristic time (units: Ma)
    temp_hist -- current temperature in history (units: deg. C)
    temp_hist_prev -- temperature from last iteration (units: deg. C)
    time -- time until end of simulation (units: Ma)
    time_prev -- previous time until end of simulation (units: Ma)
    age -- previous calculated thermochronometer age (units: Ma)
    tc_prev -- previous calculated closure temperature (units: deg. C)
    ea -- activation energy (units: J / mol)
    dtdt -- cooling rate (units: deg. C / Ma)
    a -- geometric factor (25 for a sphere, 8.7 for a planar sheet)
    d0a2 -- diffusivity at infinite temperature over domain squared (units: 1 / s)
    r -- universal gas constant (units: J / (mol K))
    """
    # Calculate diffusivity characteristic time
    tau = (r * (temp_hist + 273.15)**2.0) / (ea * dtdt)
    # Calculate new closure temperature
    tc = ea / (r * np.log(a * tau * d0a2)) - 273.15
    # Calculate new cooling age if temperature is above tc
    if temp_hist > tc:
        ratio = (tc_prev - temp_hist_prev)/(tc_prev - temp_hist_prev + temp_hist - tc)
        age = time + (time_prev - time) * ratio
    tc_prev = tc
    return age, tc, tc_prev

# Updated function
def age_predict(plot_time0=True, plot_temp_z_hist=False, n_temp_plot=1,
                temp_gradient=10.0, time_total=50.0, vz=0.5, calc_ahe=False,
                calc_zhe=False, calc_mar=False, z_max=50.0, kappa=32.0):
    """Calculate transient 1D thermal solution and predict thermochronometer ages.
    
    Keyword arguments:
    plot_time0 -- plot the initial thermal solution (default: True)
    plot_temp_z_history -- plot the temperature-depth history of the tracked particle (default: False)
    n_temp_plot -- number of temperature profiles to plot (default: 1)
    temp_gradient -- initial thermal gradient (units: deg. C / km; default: 10.0)
    time_total -- total thermal model simulation time (units: Ma; default: 50.0)
    vz -- vertical advection velocity (units: km/Ma; default: 0.5)
    calc_ahe -- calculate apatite (U-Th)/He age (default: False)
    calc_zhe -- calculate zircon (U-Th)/He age (default: False)
    calc_mar -- calculate muscovite 40Ar/39Ar age (default: False)
    """

    # THERMAL MODEL PARAMETERS
    npz = 101               # Number of depth points for temperature calculation
    npt = 401               # Number of times for temperature calculation
    high_temp = 1000.0      # Temperature to assign if tracked particle depth exceeds zmax
    
    # CLOSURE TEMPERATURE PARAMETERS
    # Apatite (U-Th)/He
    size_ahe = 100.0        # Apatite grain size [um]
    ea_ahe = 138.0e3        # Activation energy [J/mol]
    a_ahe = 25.0            # Geometry factor [25 for sphere]
    d0_ahe = 5.0e-3         # Diffusivity at infinite temperature [m2/s]
    tau_ahe = 1.0           # Initial guess for characteristic time

    # Zircon (U-Th)/He
    size_zhe = 100.0        # Zircon grain size [um]
    ea_zhe = 168.0e3        # Activation energy [J/mol]
    a_zhe = 25.0            # Geometry factor [25 for sphere]
    d0_zhe = 4.6e-5         # Diffusivity at infinite temperature [m2/s]
    tau_zhe = 1.0           # Initial guess for characteristic time

    # Muscovite Ar/Ar
    size_mar = 500.0        # Muscovite grain size [um]
    ea_mar = 183.0e3        # Activation energy [J/mol]
    a_mar = 8.7             # Geometry factor [8.7 for planar sheet]
    d0_mar = 3.3e-6         # Diffusivity at infinite temperature [m2/s]
    tau_mar = 1.0           # Initial guess for characteristic time

    # OTHER CONSTANTS
    r = 8.314               # Universal gas constant

    # Set initial thermochronometer ages
    age_ahe = time_total
    age_zhe = time_total
    age_mar = time_total

    # Convert units
    size_ahe = size_ahe / 1.0e6 / 1.0e3                                     # um -> km
    size_zhe = size_zhe / 1.0e6 / 1.0e3                                     # um -> km
    size_mar = size_mar / 1.0e6 / 1.0e3                                     # um -> km
    d0_ahe = d0_ahe * (1 / 1000.0**2.0) * (1.0e6 * 365.25 * 24.0 * 3600.0)  # m2/s -> km2/Ma
    d0_zhe = d0_zhe * (1 / 1000.0**2.0) * (1.0e6 * 365.25 * 24.0 * 3600.0)  # m2/s -> km2/Ma
    d0_mar = d0_mar * (1 / 1000.0**2.0) * (1.0e6 * 365.25 * 24.0 * 3600.0)  # m2/s -> km2/Ma

    # Calculate diffusion parameter D0/a2 for each mineral
    d0a2_ahe = d0_ahe / size_ahe**2.0    # Apatite
    d0a2_zhe = d0_zhe / size_zhe**2.0    # Zircon
    d0a2_mar = d0_mar / size_mar**2.0    # Muscovite

    # Create thermal model arrays
    z = np.linspace(0.0, z_max, npz)                    # Define depth range array
    time = np.linspace(0.0, time_total, npt)            # Define time range array
    time_ma = np.linspace(time_total, 0.0, npt)         # Define time array in Ma (time before present)
    z_hist = np.linspace(time_total*vz, 0.0, npt)       # Define z particle position history
    temp_hist = np.zeros(len(time))                     # Define initial temperature history array
    temp = np.zeros(len(z))                             # Define initial temperature array

    # Define increment for ploting output
    iout = int(float(len(time)-1)/float(n_temp_plot))

    # Create a plot window
    plt.figure(figsize=(10,7))
    
    # Upper subplot
    #plt.subplot(2, 1, 1)

    # Loop over all times and calculate temperature
    for i in range(len(time)):
        # Calculate temperature at time[i]
        temp = temp_1D(z, time[i], temp_gradient, vz, kappa)
        
        # Set the temperature history temperature to high_temp if the tracked particle
        # depth is below z_max (where temperature would be undefined)
        if z_hist[i] > max(z):
            temp_hist[i] = high_temp
        # Otherwise, store the current temperature at the depth of the tracked particle
        else:
            temp_hist[i] = temp_1D(z_hist[i], time[i], temp_gradient, vz, kappa)

        # If plotting of the initial geotherm is requested, make the plot
        if plot_time0 and i == 0:
            plt.plot(temp,-z,label=str(time_ma[i])+" Ma")

        # If the current iteration is one of the plotting increments, make the plot
        if i == iout:
            iout = iout + int(float(len(time))/float(n_temp_plot))
            plt.plot(temp,-z,label=str(time_ma[i])+" Ma")

    # Set the initial closure temperatures for the previous step to one
    tc_ahe_prev = 1.0
    tc_zhe_prev = 1.0
    tc_mar_prev = 1.0

    # Set the previous temperature to the max value in the current temperature array
    temp_hist_prev = max(temp)

    # Make 1D interpolation function for finding depth at Tc
    thist_interp = interp1d(temp_hist, z_hist)

    # Loop over all positions in the temperature history array
    for i in range(len(temp_hist)):
        # Calculate the cooling rate for the first temperature value
        if i == 0:
            dtdt = (temp_hist[i] - temp_hist[i+1]) / (time_ma[i] - time_ma[i+1])
        # Calculate the cooling rate for the last temperature value
        elif i == len(temp_hist)-1:
            dtdt = (temp_hist[i-1] - temp_hist[i]) / (time_ma[i-1] - time_ma[i])
        # Calculate the cooling rate for the the intermediate temperature values
        else:
            dtdt = (temp_hist[i-1] - temp_hist[i+1]) / (time_ma[i-1] - time_ma[i+1])

        # Ensure the cooling rate is at least 1 deg. C per 10 Ma
        dtdt = max(dtdt, 0.1 / (1.0e6 * 365.25 * 24.0 * 3600.0))

        # Calculate apatite (U-Th)/He closure temperature if requested
        if calc_ahe:
            age_ahe, tc_ahe, tc_ahe_prev = dodson(tau_ahe, temp_hist[i], temp_hist_prev, time_ma[i],
                                                  time_ma[i-1], age_ahe, tc_ahe_prev, ea_ahe, dtdt,
                                                  a_ahe, d0a2_ahe, r)

        # Calculate zircon (U-Th)/He closure temperature if requested
        if calc_zhe:
            age_zhe, tc_zhe, tc_zhe_prev = dodson(tau_zhe, temp_hist[i], temp_hist_prev, time_ma[i],
                                                  time_ma[i-1], age_zhe, tc_zhe_prev, ea_zhe, dtdt,
                                                  a_zhe, d0a2_zhe, r)

        # Calculate muscovite Ar/Ar closure temperature if requested
        if calc_mar:
            age_mar, tc_mar, tc_mar_prev = dodson(tau_mar, temp_hist[i], temp_hist_prev, time_ma[i],
                                                  time_ma[i-1], age_mar, tc_mar_prev, ea_mar, dtdt,
                                                  a_mar, d0a2_mar, r)

        # Store previous temperature in thermal history
        temp_hist_prev = temp_hist[i] 

    # Plot particle depth-temperature history if requested
    if plot_temp_z_hist:
        plt.plot(temp_hist,-z_hist,'.', color='gray', label="Thermal history")

    # Write apatite (U-Th)/He age to screen if requested
    if calc_ahe:
        print("Apatite (U-Th)/He age: {0:.2f} Ma".format(age_ahe))
        plt.text(0.05*max(temp),-0.65*max(z),("AHe age: {0:.2f} Ma").format(age_ahe))
        plt.text(0.05*max(temp),-0.7*max(z),("AHe T$_c$: {0:.1f} $^\circ$C").format(tc_ahe))
        #plt.plot([tc_ahe, tc_ahe], [min(z), -max(z)],'b-', label="Tc AHe")
        if (age_ahe <= 0.96*time_total) and (plot_temp_z_hist==True):
            plt.plot(tc_ahe, -thist_interp(tc_ahe), 'b*', label='AHe', markersize=16)

    # Write zircon (U-Th)/He age to screen if requested
    if calc_zhe:
        print("Zircon (U-Th)/He age: {0:.2f} Ma".format(age_zhe))
        plt.text(0.05*max(temp),-0.75*max(z),("ZHe age: {0:.2f} Ma").format(age_zhe))
        plt.text(0.05*max(temp),-0.8*max(z),("ZHe T$_c$: {0:.1f} $^\circ$C").format(tc_zhe))
        #plt.plot([tc_zhe, tc_zhe], [min(z), -max(z)],'g-', label="Tc ZHe")
        if (age_zhe <= 0.96*time_total) and (plot_temp_z_hist==True):
            plt.plot(tc_zhe, -thist_interp(tc_zhe), 'g*', label='ZHe', markersize=16)

    # Write muscovite Ar/Ar age to screen if requested
    if calc_mar:
        print("Muscovite Ar/Ar age: {0:.2f} Ma".format(age_mar))
        plt.text(0.05*max(temp),-0.85*max(z),("MAr age: {0:.2f} Ma").format(age_mar))
        plt.text(0.05*max(temp),-0.9*max(z),("MAr T$_c$: {0:.1f} $^\circ$C").format(tc_mar))
        #plt.plot([tc_mar, tc_mar], [min(z), -max(z)], 'r-', label="Tc MAr")
        # Plot a star for the system cooling below Tc if it does
        if (age_mar <= 0.96*time_total) and (plot_temp_z_hist==True):
            plt.plot(tc_mar, -thist_interp(tc_mar), 'r*', label='MAr', markersize=16)

    # Label axes and add title
    plt.xlabel("Temperature [$^\circ$C]")
    plt.ylabel("Depth [km]")
    plt.title("1D transient thermal solution")
    plt.axis([0, max(temp), -max(z), min(z)])
    plt.text(0.45*max(temp),-0.05*max(z),'Advection velocity: {0:.2f} km/Ma'.format(vz))

    # Display line legend
    plt.legend()