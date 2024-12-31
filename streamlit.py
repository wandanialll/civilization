import streamlit as st
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# Streamlit configuration
st.set_page_config(page_title="Population Dynamics Simulator", layout="wide")

st.title("Interactive Population Dynamics Simulator")

# Sidebar for input parameters
st.sidebar.header("Model Parameters")
r = st.sidebar.slider("Growth rate of civilians (r = b₁ - d₁)", min_value=0.001, max_value=0.05, value=0.015, step=0.001)
f_1 = st.sidebar.slider("Firing rate of bandits (f₁)", min_value=0.001, max_value=0.05, value=0.01, step=0.001)
p_1 = st.sidebar.slider("Probability of bandits hitting civilians (p₁)", min_value=0.1, max_value=1.0, value=0.3, step=0.01)
e = st.sidebar.slider("Enlistment rate of civilians to soldiers (e)", min_value=0.001, max_value=0.05, value=0.02, step=0.001)
d_2 = st.sidebar.slider("Natural death rate of soldiers (d₂)", min_value=0.001, max_value=0.05, value=0.01, step=0.001)
d_3 = st.sidebar.slider("Natural death rate of bandits (d₃)", min_value=0.001, max_value=0.05, value=0.015, step=0.001)
f_2 = st.sidebar.slider("Firing rate of soldiers (f₂)", min_value=0.001, max_value=0.05, value=0.02, step=0.001)
p_2 = st.sidebar.slider("Probability of bandits hitting soldiers (p₂)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
p_3 = st.sidebar.slider("Probability of soldiers hitting bandits (p₃)", min_value=0.1, max_value=1.0, value=0.7, step=0.01)
K = st.sidebar.slider("Carrying capacity of civilians (K)", min_value=500, max_value=5000, value=1000, step=100)
g = st.sidebar.slider("Reinforcement rate of bandits (g)", min_value=0.001, max_value=0.05, value=0.02, step=0.001)

# Sidebar for initial populations
st.sidebar.header("Initial Populations")
N0 = st.sidebar.number_input("Initial Civilians (N₀)", min_value=0, max_value=2000, value=600)
S0 = st.sidebar.number_input("Initial Soldiers (S₀)", min_value=0, max_value=1000, value=50)
B0 = st.sidebar.number_input("Initial Bandits (B₀)", min_value=0, max_value=1000, value=350)

# Sidebar for time span
st.sidebar.header("Simulation Time")
t_start = st.sidebar.number_input("Start Time (days)", min_value=0, max_value=1000, value=0)
t_end = st.sidebar.number_input("End Time (days)", min_value=t_start + 1, max_value=5000, value=1000)
t_points = st.sidebar.slider("Time Points", min_value=100, max_value=2000, value=1000)

# Define the ODEs
def odes(t, x):
    N, S, B = x
    dNdt = r * N * (1 - N / K) - f_1 * p_1 * B - e * B
    dSdt = e * B - d_2 * S - f_1 * p_2 * B
    dBdt = g * N - d_3 * B - f_2 * p_3 * S
    return [dNdt, dSdt, dBdt]

# Solve the ODEs
t_span = (t_start, t_end)
t_eval = np.linspace(*t_span, t_points)
solution = solve_ivp(
    odes,
    t_span,
    [N0, S0, B0],
    t_eval=t_eval,
)

# Extract the results and enforce non-negative populations
N = np.maximum(solution.y[0], 0)  # Civilians
S = np.maximum(solution.y[1], 0)  # Soldiers
B = np.maximum(solution.y[2], 0)  # Bandits

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(solution.t, N, label="Civilians (N)", color="blue", linewidth=2)
ax.plot(solution.t, S, label="Soldiers (S)", color="green", linewidth=2)
ax.plot(solution.t, B, label="Bandits (B)", color="red", linewidth=2)

# Add horizontal line for soldier threshold
#

# Improve plot styling
ax.set_title("Population Dynamics Over Time", fontsize=14, pad=20)
ax.set_xlabel("Time (days)", fontsize=12)
ax.set_ylabel("Population", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)

# Display final population annotations
final_day = solution.t[-1]
ax.annotate(f'Final Civilians: {int(N[-1]):,}', 
            xy=(final_day, N[-1]), 
            xytext=(-50, 10),
            textcoords='offset points')
ax.annotate(f'Final Soldiers: {int(S[-1]):,}', 
            xy=(final_day, S[-1]), 
            xytext=(-50, 10),
            textcoords='offset points')
ax.annotate(f'Final Bandits: {int(B[-1]):,}', 
            xy=(final_day, B[-1]), 
            xytext=(-50, 10),
            textcoords='offset points')

# Display the plot in Streamlit
st.pyplot(fig)
