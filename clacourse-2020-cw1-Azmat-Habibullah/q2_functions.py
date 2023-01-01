import matplotlib.pyplot as plt
import numpy as np
from cla_utils import householder_ls, cond


# compute the coefficients of the interpolating polynomial
def compute_coefficients(x_, f_, deg_poly):
    A = np.vander(x_, deg_poly+1, increasing=True)
    return A, householder_ls(A, f_)


# compute the interpolated points to plot
def calculate_interpolation(x_, f_, x_vals_, deg_poly):
    A, ai = compute_coefficients(x_, f_, deg_poly)

    # form Vandermonde matrix of values of x to plot
    vandermonde_matrix = np.vander(x_vals_, deg_poly+1, increasing=True)

    # compute the interpolated values of y to plot
    interpolated_values = vandermonde_matrix.dot(ai)

    return interpolated_values, ai


# plot interpolated polynomial
def plot(x_, f_, x_vals_, interpolated_values, deg_poly):
    plt.plot(x_, f_, "rx")
    plt.plot(x_vals_, interpolated_values, color="green")
    plt.margins(0.025)
    plt.ylim(-5, 5)
    plt.grid()
    plt.legend(["f", deg_poly])
    plt.axvline()
    plt.axhline()
    plt.show()


# compute the sensitivity to both x-values and coefficients
def analyse_sensitivity(x_, f_, deg_poly, coefficients, perturbation, interpolated_values):
    A = np.vander(x_, deg_poly + 1, increasing=True)
    sensitivity = np.zeros((11, 2))
    vandermonde = np.vander(x_, deg_poly + 1, increasing=True)
    print("Condition number", cond(A))
    for i in range(f_.size):
        perturbed_f = f_
        perturbed_f[i] += perturbation
        perturbed_values, perturbed_coefficients = calculate_interpolation(x_, perturbed_f, x_, deg_poly)
        unperturbed_values = vandermonde.dot(coefficients)
        value_error = np.linalg.norm(perturbed_values[i] - unperturbed_values[i])/perturbation
        poly_error = np.linalg.norm(perturbed_coefficients - coefficients)/perturbation
        sensitivity[i, :] = np.array([value_error, poly_error])
    return sensitivity


# plot the error graphs
def plot_sensitivitity(deg_poly, sensitivity, perturbation):
    plt.plot(np.linspace(1, 11, 11), sensitivity[:, 0], label="Value error")
    plt.plot(np.linspace(1, 11, 11), sensitivity[:, 1], label="Coefficient error")
    title = "M = " + str(deg_poly) + ", perturbation = " + str(perturbation)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()
