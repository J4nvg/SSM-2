import numpy as np
from scipy import stats
from dist.Distribution import Distribution
import time
from statsmodels.stats.weightstats import DescrStatsW
import line_profiler

"""
Previously resulted in incorrect output, yet this was due wronly chosen parameters
Now as mu = 0 and sigma = 1 the result is totally acceptable
"""


def simulate_hitting_time(dt, mu, sigma,r=1):
    x, y = 0.0, 0.0  # Starting at (0,0)
    time_elapsed = 0.0
    # Create the step distribution
    step_dist = Distribution(stats.norm(mu * dt, sigma * np.sqrt(dt)))

    while True:
        dx = step_dist.rvs()
        dy = step_dist.rvs()
        x += dx
        y += dy
        time_elapsed += dt

        # Check if |w1| >= 1
        if x ** 2 + y ** 2 >= r:
            return time_elapsed


# Simulation parameters
mu = 0
sigma = 1
nrRuns = 1000
stepSize = 1e-4

start_time = time.time()
hitting_times = np.array([simulate_hitting_time(stepSize, mu, sigma) for _ in range(nrRuns)])
print(f"Finished after {time.time() - start_time:.2f} seconds")

# Calculate empirical mean and variance of hitting times
mean_hitting_time = np.mean(hitting_times)
var_hitting_time = np.var(hitting_times)
print(f"Mean hitting time over {nrRuns} simulations = {mean_hitting_time:.4f}")
print(f"Var hitting time over {nrRuns} simulations = {var_hitting_time:.4f}")

ci = DescrStatsW(hitting_times).tconfint_mean(alpha=0.05)
print("95% CI for hitting times:", f"{ci[0]:.3f}", f"{ci[1]:.3f}")
# Compare to the theoretical result E[tau] = 0.5
print(f"Difference from 0.5 = {abs(mean_hitting_time - 0.5):.4f}")



def oneC(dt, mu, sigma,r_start,r_inner,r_outer):
    x = r_start / np.sqrt(2) # Start at top-right as in image
    y = r_start / np.sqrt(2)
    time_elapsed = 0.0
    # Create the step distribution
    step_dist = Distribution(stats.norm(mu * dt, sigma * np.sqrt(dt)))

    while True:
        dx = step_dist.rvs()
        dy = step_dist.rvs()
        x += dx
        y += dy
        time_elapsed += dt

        if x ** 2 + y ** 2 >= r_outer**2:
            return time_elapsed,1
        if x ** 2 + y ** 2 <= r_inner**2:
            return time_elapsed,0



nrRuns = 1000
stepSize = 1e-4
#Predefined n
n = 10
def r(i):
    radi = 4**(-1) + (4*10)**(i/n)
    return radi


# for i_star in range(2, n-1):
#     inner_times = np.zeros(nrRuns)
#     outer_times = np.zeros(nrRuns)
#     for i in range(nrRuns):
#         tau, boundary = oneC(stepSize, 0, 1, r(i_star), r(i_star - 1), r(i_star + 1))
#         if boundary == 0:
#             inner_times[i] = tau
#         else:
#             outer_times[i] = tau
#
#
#     Eone = np.mean(inner_times[inner_times < outer_times])
#     Prob = np.mean(inner_times < outer_times)
#     Etwo = np.mean(outer_times[outer_times > inner_times])
#
#     print(f"i* = {i_star}:")
#     print("E[τ_inner | τ_inner < τ_outer] =", Eone)
#     print("P[τ_outer < τ_inner] =", Prob)
#     print("E[τ_outer | τ_outer < τ_inner] =", Etwo)

for i_star in range(2, n-1):
    print(f"==== [{i_star}] ====")
    hitting_times = np.zeros(nrRuns)
    boundaries = np.zeros(nrRuns, dtype=int)

    for i in range(nrRuns):
        tau, boundary = oneC(stepSize, 0, 1, r(i_star), r(i_star - 1), r(i_star + 1))
        hitting_times[i] = tau
        boundaries[i] = boundary

    # Create boolean masks for inner and outer boundary hits.
    inner_mask = (boundaries == 0)
    outer_mask = (boundaries == 1)

    # Calculate the conditional expectations using the masks.
    E_inner = np.mean(hitting_times[inner_mask])
    print("E[τ_inner | τ_inner < τ_outer] =", E_inner)

    E_outer = np.mean(hitting_times[outer_mask])
    print("E[τ_outer | τ_inner < t_outer] =", E_outer)

    # Calculate the probability of hitting the outer boundary.
    P_outer = np.mean(outer_mask)  # Booleans are interpreted as 0 (False) or 1 (True)
    print("P[τ_outer < τ_inner] =", P_outer)