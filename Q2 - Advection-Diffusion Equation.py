"""
Question 2: Advection-Diffusion Equation

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
x = np.linspace(0, 100, num = 1000) # Domain of the function
F_ini = np.copy(x) # Initial condition of the function [F(x,0) = x]
F_D1 = np.copy(F_ini) # Copy of the initial function, which will be modified in the Lax-Friedrich loop
F_D2 = np.copy(F_ini)
u = -0.1 # Given in the problem
c = 0.5 * u * delta_t/delta_x # Courant-Friedrich-Lewy number


# Diffusion constants
D1 = 5
D2 = 100


# Value of Beta for the implicit method
B1 = D1 * delta_t / (delta_x)**2
B2 = D2 * delta_t / (delta_x)**2



# Setting up the plot
plt.ion()
fig, ax = plt.subplots(1,2, sharey=True)

# First plot (using D1)
f_d1, = ax[0].plot(x, F_D1, color='fuchsia', linewidth=4.0)
ax[0].set_title('Diffusion coefficient = ' + str(D1))

# Second plot (using D2)
f_d2, = ax[1].plot(x, F_D2, color='orange', linewidth=4.0)
ax[1].set_title('Diffusion coefficient = ' + str(D2)) 

fig.canvas.draw()



while counter < steps:
# D1 loop
    # Implicit method
    A1 = np.eye(j) * (1 + 2 * B1) + np.eye(j, k=1) * (-B1) + np.eye(j, k=-1) * (-B1)
    
    # Ensure v[0] stays fixed at all times
    A1[0][0] = 1
    A1[0][1] = 0
    # Send F_j+1 - F_j and F_j - F_j-1 to zero for right-most boundary
    A1[-1][-1] = 1 + B1
    # solve to recover F
    F_D1 = np.linalg.solve(A1, F_D1)
    
    # Avection using Lax-Friedriech
    F_D1[1:j-1] = 0.5 * (F_D1[2:] + F_D1[:j-2]) -  c/2 * (F_D1[2:] - F_D1[:j-2])
    
    
# D2 loop
    # Implicit method
    A2 = np.eye(j) * (1 + 2 * B2) + np.eye(j, k=1) * (-B2) + np.eye(j, k=-1) * (-B2)
    # Ensure v[0] stays fixed at all times
    A2[0][0] = 1
    A2[0][1] = 0
    # Send F_j+1 - F_j and F_j - F_j-1 to zero for right-most boundary
    A2[-1][-1] = 1 + B2
    # solve to recover F
    F_D2 = np.linalg.solve(A2, F_D2)
    
    # Avection using Lax-Friedriech
    F_D2[1:j-1] = 0.5 * (F_D2[2:] + F_D2[:j-2]) -  c/2 * (F_D2[2:] - F_D2[:j-2])
    
    
    # Update the plot with new values of F for both values of D
    f_d1.set_ydata(F_D1)
    f_d2.set_ydata(F_D2)
    fig.canvas.draw()
    plt.pause(0.001)
    counter += 1