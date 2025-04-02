from scipy.optimize import root_scalar
import numpy as np

def original_equation(e_r):
    # Correcting to the actual equation from your screenshot: 9 - 5 = 100 - 99 * e^(-r) - 2 * e^(-r * 0.5)
    return 9 - 5 - (100 - 99 * (1 / e_r) - 2 * (1 / np.sqrt(e_r)))

# Solve the equation for e^r using root_scalar
solution = root_scalar(original_equation, method='brentq', bracket=[0.1, 10])

if solution.converged:
    e_r = solution.root
    print(f"The value of e^r is approximately: {e_r:.6f}")
else:
    print("The solver did not converge.")
