import numpy as np

# Parameters
L = 5
Nx = 501
delx = L / (Nx - 1)
x = np.linspace(0, 5, Nx)
g = 9.81
lambda_cfl = 0.8
T = 0

# Initial time step calculation
delt = (lambda_cfl * delx) / np.sqrt(g * 2)
U1 = np.zeros(Nx)
U2 = np.zeros(Nx)

# Initial conditions
for i in range(Nx):
    U1[i] = 1
    U2[i] = 0
    if x[i] < 2.5:
        U1[i] = 2

i = 0
sr = 0
T += delt

while T <= 1:
    i += 1
    U1_new = U1.copy()
    U2_new = U2.copy()

    for j in range(1, Nx-1):
        U1_new[j] = 0.5 * (U1[j + 1] + U1[j - 1]) - 0.5 * delt * (U2[j + 1] - U2[j - 1]) / delx
        U2_new[j] = 0.5 * (U2[j + 1] + U2[j - 1]) - 0.5 * delt * (
            (U2[j + 1] * U2[j + 1] / U1[j + 1] + 0.5 * g * U1[j + 1] * U1[j + 1]) - 
            (U2[j - 1] * U2[j - 1] / U1[j - 1] + 0.5 * g * U1[j - 1] * U1[j - 1])
        ) / delx

    # Boundary conditions
    U1_new[0] = 2
    U1_new[-1] = 1
    U2_new[0] = 0
    U2_new[-1] = 0

    U1 = U1_new
    U2 = U2_new

    sr = 0
    for j in range(Nx):
        sr_temp = abs(U2[j] / U1[j]) + np.sqrt(g * U1[j])
        sr = max(sr_temp, sr)
    
    delt = lambda_cfl * delx / sr
    T += delt

h = U1
print(U2)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(x, U1)
plt.title('Water Height (U1)')
plt.xlabel('x')
plt.ylabel('U1')

plt.subplot(2, 1, 2)
plt.plot(x, U2)
plt.title('Velocity (U2)')
plt.xlabel('x')
plt.ylabel('U2')

plt.tight_layout()
plt.show()
