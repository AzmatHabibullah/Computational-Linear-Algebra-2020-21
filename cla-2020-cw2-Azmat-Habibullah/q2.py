from q1 import *
from cla_utils import LU_inplace, solve_L, solve_U, solve_LU
import timeit
import matplotlib.pyplot as plt


def check_extrema_constant(m):
    """
    Plot the ratios of values in the top right and bottom left corners
    in each iteration of A in the form of question to see that A_{1, m} and
    A_{m, 1} are constant, as expected, for part c
    @param m: size of matrix
    """
    c = np.random.normal(0, m**2)**2
    A = construct_A(m, 1 + 2*c, -c)
    # perform LU_inplace
    top_corner, bottom_corner = LU_inplace(A, return_corners=True)
    top_change = np.array([top_corner[i+1]/top_corner[i] for i in range(m-1)])
    bottom_change = np.array([bottom_corner[i+1]/bottom_corner[i] for i in range(m-1)])
    # verify that the top right and bottom left entries are unchanged
    assert np.linalg.norm(top_change[:-1] - 1) < 10e-10
    assert np.linalg.norm(bottom_change[:-1] - 1) < 10e-10
    vals = np.linspace(1, m-1, m-1)
    plt.plot(vals, top_change, label="a{1, m+1}/a_{1, m}")
    plt.plot(vals, bottom_change, label="a_{m+1, 1}/a_{m, 1}")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Ratio of adjacent entries")
    plt.title("Ratio of extreme entries of A in each step in LU factorisation")
    plt.show()


def solve_Ax_b(c, d, b):
    """
    Algorithm for part f: solve Ax = b for given system
    @param c: diagonal entries of matrix
    @param d: sub/superdiagonal entries of matrix
    @param b: vector to solve for
    @return: solution x
    """
    # initialise variables
    m = b.size
    u1 = np.zeros(m, dtype=complex)
    u2 = np.zeros(m, dtype=complex)
    u1[0] = 1
    u2[m-1] = 1
    B = np.zeros((b.size, 3), dtype=complex)
    B[:, 0] = b
    B[:, 1] = u1
    B[:, 2] = u2
    solved = solve_tridiag_LU_inplace(c, d, B)
    # execute algorithm
    a = solved[:, 0]
    mu = solved[:, 1]
    nu = solved[:, 2]
    alpha = 1 / (1 + d*mu[m-1])
    big_denom = 1 + d*nu[0] - alpha * d**2*mu[0] * nu[m-1]
    x = a - alpha * d*a[m-1]*mu - 1/big_denom * (d*nu*a[0] - alpha*d**2 *mu[0]*a[m-1]*nu
                                                        - alpha*d**2*a[0]*nu[m-1]*mu
                                                        + alpha**2 * d**3 * nu[m-1]*mu[0]*a[m-1]*mu)
    return x


def compare_solvers(max_m, step_size=20):
    """
    Compare LU solver vs new implementation for part f
    @param max_m: maximum matrix dimension to test speed up to
    @param step_size: steps size for testing m
    @return: time taken to compute LU solutions, time taken to compute new algorithm solutions
    """
    # initialise variables
    LU_times = np.zeros(max_m//step_size)
    algo_times = np.zeros(max_m//step_size)
    for m in range(step_size, max_m+1, step_size):
        c = np.random.normal()
        d = np.random.normal()
        A = construct_A(m, c, d, return_all=False)
        b = np.random.rand(m)
        # time algorithms for random values
        LU_times[m//step_size-1] = timeit.timeit(lambda: solve_LU(A, b.reshape(m, 1)), number=1)
        algo_times[m//step_size-1] = timeit.timeit(lambda: solve_Ax_b(c, d, b), number=1)
    # plot results
    iterates = np.arange(step_size, max_m+1, step_size)
    plt.plot(iterates, LU_times, label="LU times")
    plt.plot(iterates, algo_times, label="2d method times")
    plt.legend()
    plt.title("LU solver vs new implementation")
    plt.yscale("log")
    plt.xlabel("Matrix degree")
    plt.ylabel("Time to solve")
    plt.show()
    return LU_times, algo_times


def construct_A(m, c, d, return_all=False):
    """
    Construct A of the necessary form
    @param m: dimensions of matrix A
    @param c: diagonal entries
    @param d: sub/superdiagonal entries
    @param return_all: obvious
    @return: A
    """
    T = construct_tridiag(c, d, m)
    # initialise vectors
    u1 = np.zeros(m, dtype=complex)
    u2 = np.zeros(m, dtype=complex)
    v1 = np.zeros(m, dtype=complex)
    v2 = np.zeros(m, dtype=complex)
    u1[0] = 1
    v1[m-1] = d
    u2[m-1] = 1
    v2[0] = d
    # compute A
    A = T + np.outer(u1, v1) + np.outer(u2, v2)
    if return_all:
        return A, T, u1, u2, v1, v2
    return A


def solve_pde(M=100,N=3, u0 = None, w0 = None, deltat=1/100, timesteps_to_plot=None, timesteps_to_save = None, filename=None):
    """
    Solve the pde in the 2g
    @param M: M in question
    @param N: N in question
    @param u0: initial conditions for u0(x). default sin(pix)
    @param w0: initial conditions for w0(x)
    @param deltat: time interval
    @param timesteps_to_plot: obvious
    @param timesteps_to_save: obvious
    @param filename: filename to save
    """
    if timesteps_to_plot is not None:
        u_to_plot = np.zeros((M, timesteps_to_plot.size))
    if timesteps_to_save is not None:
        u_to_save = np.zeros((M, timesteps_to_save.size))

    deltax = 1 / M
    C1 = deltat ** 2 / (4 * deltax ** 2)

    # initial conditions
    if u0 is None:
        ui = np.sin(np.linspace(0, 1, M)*np.pi)
    else:
        ui = u0
    if w0 is None:
        wi = np.zeros(M)
    else:
        wi = w0

    # iteratively solve system
    for i in range(N):
        # compute w
        b = wi + deltat / (deltax ** 2) * (np.roll(ui, -1) - 2 * ui + np.roll(ui, 1)) + C1 * (
                    np.roll(wi, -1) - 2 * wi + np.roll(wi, 1))
        wi_minus_1 = 1.0*wi
        wi = solve_Ax_b(1 + 2 * C1, -C1, b)
        # back substitute for u
        ui = ui + deltat / 2 * (wi + wi_minus_1)
        # handle plotting/saving
        if timesteps_to_plot is not None and i in timesteps_to_plot:
            u_to_plot[:, np.where(timesteps_to_plot == i)[0][0]] = ui
        if timesteps_to_save is not None and i in timesteps_to_save:
            u_to_save[:, np.where(timesteps_to_save == i)[0][0]] = ui
    # plot timesteps if necessary
    if timesteps_to_plot is not None:
        for t in range(timesteps_to_plot.size):
            plt.plot(np.linspace(0, 1, M), u_to_plot[:, t], label="t")
        legend = ["t = " + str(i) for i in timesteps_to_plot]
        plt.legend(legend)
        plt.xlabel("x")
        plt.ylabel("u(x, t)")
        plt.title("1D wave equation discretisation")
        plt.show()
    # save timesteps if necessary
    if timesteps_to_save is not None:
        title = "2g plots/" + filename
        np.save(title, u_to_save)


if __name__ == '__main__':
    # part f
    # --- test how long our algorithm takes to run compared to LU_inplace
    # --- degree 1000 (in report) takes some time to run (~1 minute)
    # --- for speed, decrease degree or increase step_size
    _, _ = compare_solvers(1000, step_size=20)

    # part g
    solve_pde(M=1000, N=100, timesteps_to_plot=np.array([10, 30, 50, 70, 90]))