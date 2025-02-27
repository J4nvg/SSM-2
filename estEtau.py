import numpy as np
from scipy import stats
from dist.Distribution import Distribution
import time
from statsmodels.stats.weightstats import DescrStatsW

"""
Big chance the code has some huge inefficiencies as it
takes a long time before it is done with a high amount of sims
"""


def simulate_hitting_times(nrRuns, dt, mu, sigma):
    hitting_times = np.zeros(nrRuns)
    step_dist = Distribution(stats.norm(mu * dt, sigma * np.sqrt(dt)))
    for i in range(nrRuns):
        x, y = 0.0, 0.0  # Starting at (0,0)
        time = 0.0
        while True:
            dx = step_dist.rvs()
            dy = step_dist.rvs()
            x += dx
            y += dy
            time += dt

            # Check if |w1| >= 1
            if x ** 2 + y ** 2 >= 1.0:
                hitting_times[i] = time
                break
    return hitting_times


mu = 1
sigma = 0.5
nrRuns = 1000
stepSize = 1e-4
startTime = time.time()
times = simulate_hitting_times(nrRuns, stepSize, mu, sigma)
print(f"finished after {time.time() - startTime:.2f} seconds")
# Empirical mean and var hitting time
mean_hitting_time = np.mean(times)
var_hitting_time = np.var(times)
print(f"Mean(X) hitting time over {nrRuns} simulations = {mean_hitting_time:.4f}")
print(f"Var(X) hitting time over {nrRuns} simulations = {var_hitting_time:.4f}")
ci = DescrStatsW(times).tconfint_mean(alpha=0.05)
print("95% CI for profits:",f"{ci[0]:.3f}",f"{ci[1]:.3f}", "halfWidth:",f"{(mean_hitting_time - ci[0]):.3f}")

# Compare to the theoretical result E[tau] = 0.5
print(f"Difference from 0.5 = {abs(mean_hitting_time - 0.5):.4f}")
