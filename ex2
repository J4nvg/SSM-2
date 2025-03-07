import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Circle


# 2.a)

T = 100
M = 20000
r1 = 1
r2 = 3

# hard reflection
def hardReflection(T, M, r1, r2, x0, y0):
    dt = T/M
    normDist = stats.norm(0, np.sqrt(dt))
    incrementsX = normDist.rvs(M)
    incrementsY = normDist.rvs(M)
    
    # I choose (0,2) as starting point, so the distance from center is 2
    
    distance = 0 # distance from center
    for i in range(M):
        x = np.cumsum(incrementsX)
        y = np.cumsum(incrementsY)
        distance = np.sqrt(x**2 + y**2)[i]
        
        if (r1 > distance) or (distance > r2):
            hardReflection(T, M-i+1, r1, r2, x, y)
    
    sbmX = np.append([x0],np.cumsum(incrementsX))
    sbmY = np.append([y0],np.cumsum(incrementsY))
    circle1 = Circle((0,0), r1, fill=False, color="red")
    circle2 = Circle((0,0), r2, fill=False, color="red")
    fig, ax = plt.subplots()
    ax.set_xlim((-4, 4))
    ax.set_ylim((-4, 4))
    fig = plt.plot(sbmX, sbmY)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    plt.show()
    
    return sbmX, sbmY

hardReflection(T, M, r1, r2, 0, 2)
