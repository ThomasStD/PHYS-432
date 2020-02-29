"""
Question 1: Advection Equation

@author: Thomas St-Denis
Friday 28th, January 2020
"""

# Importing the relevant packages
import numpy as np
import matplotlib.pyplot as plt



# Set up the initial parameters
steps = 1000 # Number of times the loop is computed
j = 1000 # Number of time steps computed in the loop
delta_t = 0.01 # Resolution of the time intervals
delta_x = 0.001 # Resolution of the position intervals
counter = 0 # Initializing the step value for the loops



# Definition of the function F and its parameters
x = np.linspace(0, 1, num = 1000) # Domain of the function
F_ini = np.copy(x) # Initial condition of the function [F(x,0) = x]
F_ftcs = np.copy(F_ini) # Copy of the initial function, which will be modified in the FTCS loop 
F_lax = np.copy(F_ini) # Copy of the initial function, which will be modified in the Lax-Friedrich loop 
u = -0.1 # Given in the problem
c = 0.5 * u * delta_t/delta_x # Courant-Friedrich-Lewy number



# Setting up the plot
plt.ion()
fig, ax = plt.subplots(1,2, sharey=True)

# FTCS plot
FTCS, = ax[0].plot(x, F_ftcs, color='fuchsia', linewidth=4.0, label ='FTCS') # FTCS function to plot and update
ax[0].set_title('FTCS') # Title for the FTCS subplot

# Lax-Friedrich plot
LAX, = ax[1].plot(x, F_lax, color='orange', linewidth=4.0)
ax[1].set_title('Lax-Friedrich')

# Exact solutions used for comparison (found using Mathematica) -> JUST FOR FUN
Exactsol0, = ax[0].plot(x,F_ini, '--', color='black', linewidth=1, label='Exact sol')
Exactsol1, = ax[1].plot(x,F_ini, '--', color='black', linewidth=1, label='Exact sol')

# Legends for both plots
ax[0].legend(["FTCS","Exact sol"], loc='upper left')
ax[1].legend(["Lax-Friedrich","Exact sol"], loc='upper left')
fig.canvas.draw()



while counter < steps:
# FTCS loop
    # Update the value of F_j to move from time steps n to n+1
    F_ftcs[1:j-1] = F_ftcs[1:j-1] - c * (F_ftcs[2:] - F_ftcs[:j-2])

#Lax-Friedrich loop    
    # Update the value of F_j to move from time steps n to n+1
    F_lax[1:j-1] = 0.5 * (F_lax[2:] + F_lax[:j-2]) -  c/2 * (F_lax[2:] - F_lax[:j-2])
    
    # Update the plot with new values of F_ftcs and F_lax
    FTCS.set_ydata(F_ftcs)
    LAX.set_ydata(F_lax)
    fig.canvas.draw()
    plt.pause(0.001)
    counter += 1