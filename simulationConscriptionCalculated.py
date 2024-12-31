from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# Define the ODEs
def odes(t, x):
    # Model parameters (unchanged)
    r = 0.015
    f_1 = 0.01
    p_1 = 0.9
    e = 0.02
    d_2 = 0.01
    d_3 = 0.015
    f_2 = 0.02
    p_2 = 0.05
    p_3 = 0.7
    K = 1000
    g = 0.02

    N, S, B = x

    # ODEs
    dNdt = r * N * (1 - N / K) - f_1 * p_1 * B - e * B 
    dSdt = e * B - d_2 * S - f_1 * p_2 * B 
    dBdt = g * N - d_3 * B - f_2 * p_3 * S

    return [dNdt, dSdt, dBdt]

# Variables
N = 600
S = 50
B = 350

# Time span (in days)
t_span = (0, 1000)
t_eval = np.linspace(*t_span, 1000)  # Time points for evaluation

# Solve the ODEs
solution = solve_ivp(
    odes,
    t_span,
    [N,S,B],
    t_eval=t_eval,
    #method="RK45",  # Runge-Kutta method
)


# Extract the results and enforce non-negative populations
N = np.maximum(solution.y[0], 0)  # Civilians
S = np.maximum(solution.y[1], 0)  # Soldiers
B = np.maximum(solution.y[2], 0)  # Bandits

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(solution.t, N, label="Civilians (N)", color="blue", linewidth=2)
plt.plot(solution.t, S, label="Soldiers (S)", color="green", linewidth=2)
plt.plot(solution.t, B, label="Bandits (B)", color="red", linewidth=2)

# Improve plot styling
plt.title("Population Dynamics Over Time", fontsize=14, pad=20)
plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("Population", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Add final population annotations
final_day = solution.t[-1]
plt.annotate(f'Final Civilians: {int(N[-1]):,}', 
            xy=(final_day, N[-1]), 
            xytext=(-50, 10),
            textcoords='offset points')
plt.annotate(f'Final Soldiers: {int(S[-1]):,}', 
            xy=(final_day, S[-1]), 
            xytext=(-50, 10),
            textcoords='offset points')
plt.annotate(f'Final Bandits: {int(B[-1]):,}', 
            xy=(final_day, B[-1]), 
            xytext=(-50, 10),
            textcoords='offset points')

plt.show()