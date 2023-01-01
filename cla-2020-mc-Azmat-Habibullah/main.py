import numpy as np
import matplotlib.pyplot as plt
from plot import plot_regularisation_effect, compute_errors, plot_fourier, plot_condition_numbers, plot_c_choices_N


def runge(x):
    return 1/(1 + 25*x**2)


def f2(x):
    return 1/(10 - 9*x)


if __name__ == '__main__':
    # code for plots in report are here, with parameter values (n, precision, resolution) lowered for speed concerns
    # error comparison for 1000 equispaced points in [-1, 1]
    compute_errors(300, 1000, runge, n=50, show_plots=True)
    # error comparison for 1000 Chebyshev points in [-1, 1]
    compute_errors(300, 1000, runge, n=50, show_plots=True, cheb=True)

    # fourier extension error plot
    M = 200
    x = np.cos(np.linspace(0, M, M+1) * np.pi/M)
    s = np.linspace(-2, 2, 101)
    f = 1 / (10 - 9 * x)
    compute_errors(501, 1000, f2, n=50, show_plots=True, cheb=True, fourier=True)

    # fourier extension plots for n = 40
    x_cheb = np.cos(np.linspace(0, M, M+1) * np.pi/M)
    plt.plot(s, 1 / (10 - 9 * s), 'x')
    plot_fourier(x, f, 40, s, arnoldi=True)

    plt.plot(s, 1 / (10 - 9 * s), 'x')
    plot_fourier(x, f, 40, s)

    # Gram matrix condition number plot
    _ = plot_condition_numbers(50)

    # preconditioning plots
    x = np.linspace(-1, 1, 100)
    z = np.e ** (1.0j * np.pi * x / 2)
    plot_c_choices_N(z, "Transformed (z)", max_N=50, resolution=1/10, upper_bound=10000)
    plot_c_choices_N(x, "Nontransformed (x)", max_N=50, resolution=1/10, upper_bound=10000)

    # truncated SVD regularisation plots
    plot_regularisation_effect(n=50)
