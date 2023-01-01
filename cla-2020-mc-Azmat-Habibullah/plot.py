import matplotlib.pyplot as plt
import numpy as np
from svd import make_gram
from fit import polyval, polyvalA, polyfitA, polyfit, compute_interpolation


def compute_errors(M, precision, f_to_fit, show_plots=None, n=50, cheb=False, fourier=False, truncation=0):
    """
    Computes errors for approximation up to degree n
    :param M: M in least squares problem (number of data points)
    :param precision: number of sample points to compare error to
    :param f_to_fit: function to approximation
    :param show_plots: if true, show plots
    :param n: maximum degree
    :param cheb: if true, use chebyshev nodes to interpolate. if false, use equispaced points
    :param fourier: if true, approximate via fourier series - this is least squares, not interpolation
    :param truncation: if non-zero, truncate this many singular values
    :return:
    """
    # initialise variables
    vander_errors = np.zeros(n)
    arnoldi_errors = np.zeros(n)
    if fourier:
        s = np.linspace(-2, 2, 2*precision + 1)
        true_vals = f_to_fit(s[precision//2:3*precision//2 + 1])
    else:
        s = np.linspace(-1, 1, precision+1)
        true_vals = f_to_fit(s)

    for k in range(1, n+1):
        if fourier:
            # least squares
            if cheb:
                x = np.cos(np.linspace(0, M, M) * np.pi / M)
            else:
                x = 2 * np.linspace(0, M, M + 1) / M - 1
        else:
            # interpolation
            if cheb:
                x = np.cos(np.linspace(0, k, k+1) * np.pi/k)
            else:
                x = 2*np.linspace(0, k, k+1)/k - 1
        f = f_to_fit(x)

        # compute interpolation
        polyval_interpolation = compute_interpolation(x, f, k, s, arnoldi=False, fourier=fourier, truncation=truncation)
        polyvalA_interpolation = compute_interpolation(x, f, k, s, arnoldi=True, fourier=fourier, truncation=truncation)

        # calculate errors, noting the larger s space for fourier extensions
        if fourier:
            vander_errors[k - 1] = np.linalg.norm(polyval_interpolation[precision//2:3*precision//2 + 1] - true_vals)
            arnoldi_errors[k - 1] = np.linalg.norm(polyvalA_interpolation[precision//2:3*precision//2 + 1] - true_vals)
        else:
            vander_errors[k-1] = np.linalg.norm(polyval_interpolation - true_vals)
            arnoldi_errors[k-1] = np.linalg.norm(polyvalA_interpolation - true_vals)

    # handle plots
    if show_plots is not None:
        plot_errors(vander_errors, "Vandermonde error", n)
        plot_errors(arnoldi_errors, "Vandermonde + Arnoldi error", n)
    return vander_errors, arnoldi_errors


def plot_fourier(x, f, n, s, arnoldi=False, truncation=0):
    """
    Plot an individual function's Fourier approximation
    :param x: data points
    :param f: function values
    :param n: degree of approximation
    :param s: evaluation points
    :param arnoldi: if true, use Arnoldi method
    :param truncation: if non-zero, truncate this many singular values
    :return:
    """
    z = np.e**(1j*np.pi*x/2)
    if arnoldi:
        Q, H, d = polyfitA(np.diag(z), f, n, fourier=True, truncation=truncation)
        plt.plot(s, polyvalA(d, H, np.e**(1j*np.pi*s/2), fourier=True))
        plot_function(s, polyvalA(d, H, np.e**(1j*np.pi*s/2), fourier=True), title="Arnoldi Fourier extension")
    else:
        c = polyfit(z, f, n, fourier=True)
        plt.plot(s, polyval(c, np.e**(1j*np.pi*s/2), fourier=True))
        plot_function(s, polyval(c, np.e**(1j*np.pi*s/2), fourier=True), title="Vandermonde Fourier extension")


def plot_errors(errors, title, n):
    """
    Plot errors computed in compute_errors
    :param errors: errors
    :param title: title of plot
    :param n: degree of approximation
    """
    plt.plot(np.linspace(1, n, n), errors)
    plt.yscale("log")
    plt.ylabel("||f - p||")
    plt.title(title)
    plt.ylim(10 ** -15)
    plt.xlabel("n")
    plt.show()


def plot_function(x, y, title):
    """
    Function to ease with plotting
    :param x: x values
    :param y: y values
    :param title: plot title
    """
    plt.plot(x, y)
    plt.title(title)
    plt.ylabel("p(x)")
    plt.xlabel("x")
    plt.ylim(-1.3, 1.3)
    plt.show()


def plot_condition_numbers(max_n):
    conditions = np.zeros(max_n)
    for i in range(1, max_n+1):
        A = make_gram(i)
        conditions[i-1] = np.linalg.cond(A)
    plt.plot(np.linspace(1, max_n, max_n), conditions)
    plt.title("Condition numbers for NxN Gram matrices")
    plt.xlabel("N")
    plt.ylabel("Condition nummber")
    plt.yscale("log")
    plt.show()
    return conditions


def f2(x):
    return 1/(10 - 9*x)


def plot_regularisation_effect(n=50, f=f2):
    """
    Compute and plot the regularisation effect for truncation of 0, 20, 50 and 100 singular values
    :param n: maximum degree
    :param f: function to approximate
    """
    vander0, arnoldi0 = compute_errors(501, 100, f, n=n, cheb=True, fourier=True, truncation=0)
    vander20, arnoldi20 = compute_errors(501, 100, f, n=n, cheb=True, fourier=True, truncation=20)
    vander50, arnoldi50 = compute_errors(501, 100, f, n=n, cheb=True, fourier=True, truncation=50)
    vander100, arnoldi100 = compute_errors(501, 100, f, n=n, cheb=True, fourier=True, truncation=100)

    values = np.linspace(1, n, n)
    plt.plot(values, vander0, label="No truncation")
    plt.plot(values, vander20, label="20 truncations")
    plt.plot(values, vander50, label="50 truncations")
    plt.plot(values, vander100, label="100 truncations")
    plt.legend()
    plt.title("Change in errors with truncations for Vandermonde")
    plt.xlabel("n")
    plt.ylabel("||f - p||")
    plt.yscale("log")
    plt.show()

    plt.plot(values, arnoldi0, label="No truncation")
    plt.plot(values, arnoldi20, label="20 truncations")
    plt.plot(values, arnoldi50, label="50 truncations")
    plt.plot(values, arnoldi100, label="100 truncations")
    plt.legend()
    plt.title("Change in errors with truncations for Arnoldi")
    plt.xlabel("n")
    plt.yscale("log")
    plt.ylabel("||f - p||")
    plt.show()


def plot_c_choices_N(x, title, max_N=100, upper_bound=10000, resolution=1/1000):
    """
    Attempt to find a good c for a preconditioner M = cU
    :param x: interpolation points
    :param title: plot title
    :param max_N: maximum dimension of matrix
    :param upper_bound: upper bound for condition numbers to plot
    :param resolution: resolution of c
    """
    best_cs = np.zeros(max_N)
    value = np.zeros(max_N)
    for n in range(1, max_N):
        A = np.vander(x, n+1, increasing=True)
        best_cs[n], value[n] = best_c(A, return_best=True, upper_bound=upper_bound, resolution=resolution, right=True)
    # plot on one set of axes
    fig, ax1 = plt.subplots()
    plot_vals = np.linspace(1, max_N, max_N)
    color = 'tab:red'
    ax1.set_xlabel('N')
    ax1.set_ylabel('Best c', color=color)
    ax1.plot(plot_vals, best_cs, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('||I - AM^-1||', color=color)
    ax2.plot(plot_vals, value, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title(title)

    fig.tight_layout()
    plt.show()


def best_c(A, return_best=False, right=False, upper_bound=991, resolution=1/1000):
    """
    For given matrix A, determine the c that minimises the error |I - M^{-1}A|
    with M = cU, where U is the upper triangular part of A
    @param A: matrix to determine c for
    @param resolution: if true, return the best value
    @param return_errors: 2x991 array: if true, return all of the errors for various c.
                          First row is values of c, second is the error
    @param upper_bound: upper bound to test
    @return: the value of c in [0.1, 10] which minimises the error. Most errors were
             experimenally determined to fall within this range hence it has been choosed
             for efficiency reasons.
    """
    m, k = A.shape
    precision = int(1//resolution)
    errors = np.zeros((2, precision))
    cs = np.linspace(0.1, upper_bound, precision)
    errors[0, :] = cs
    for i in range(precision):
        M = cs[i] * np.triu(A)
        if right:
            y = np.linalg.norm(np.eye(m, k) - A.dot(np.linalg.inv(M[:k])))
        else:
            y = np.linalg.norm(np.eye(m) - np.linalg.inv(M).dot(A))
        errors[1, i] = y
    best_c = errors[0, np.argmin(errors[1, :])]
    if return_best:
        return best_c, errors[1, np.argmin(errors[1, :])]
    return best_c
