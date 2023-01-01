from q3_functions import *
from cla_utils import cond
import numpy as np

np.set_printoptions(linewidth=320)

M = 1000 # even and odd M change plots!
x = np.arange(-1 + 1/(2*M), 1, 1/M)


if __name__ == "__main__":
    # part a
    p = 5
    V = np.vander(x, p+1, increasing=True)
    hQ, hR = householder_qr(V)
    plot_part_a(x, hQ)

    # part b

    # set of values of p to plot for
    p_vals = [5, 10, 15, 25, 40, 45, 50, 100]
    condition_numbers = {}

    # calculate (the largest) V once and truncate as needed rather than recalculating wastefully
    V = np.vander(x, np.max(p_vals), increasing=True)
    for p in p_vals:
        relevant_part_of_V = V[:, :p+1]
        condition_numbers[p] = cond(relevant_part_of_V)
        householder_Q, classical_Q, modified_Q = reduced_Qs(relevant_part_of_V, p)
        plot_all(x, p, householder_Q, classical_Q, modified_Q)
    print(condition_numbers)

