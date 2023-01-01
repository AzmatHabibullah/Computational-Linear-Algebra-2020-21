from numpy import *
from q2_functions import *


if __name__ == "__main__":
    x = arange(-1, 1.1, 0.2)
    f = 0.0 * x
    f[3:6] = 1

    precision = 250
    perturbations = [10**-2, 10**-4, 10**-6]

    # part c
    print("\nm = 10")
    x_vals = linspace(-1, 1.1, precision)
    deg_10_values, deg_10_coefficients = calculate_interpolation(x, f, x_vals, 10)
    print(np.around(deg_10_coefficients, 3))
    plot(x, f, x_vals, deg_10_values, 10)

    # part d
    for perturbation in perturbations:
        sens_10 = analyse_sensitivity(x, f, 10, deg_10_coefficients, perturbation, deg_10_values)
        plot_sensitivitity(10, sens_10, perturbation)

    # part e
    print("\nm = 7")
    deg_7_values, deg_7_coefficients = calculate_interpolation(x, f, x_vals, 7)
    print(np.around(deg_7_coefficients, 3))
    plot(x, f, x_vals, deg_7_values, 7)

    # part f
    for perturbation in perturbations:
        print(perturbation)
        sens_7 = analyse_sensitivity(x, f, 7, deg_7_coefficients, perturbation, deg_7_values)
        plot_sensitivitity(7, sens_7, perturbation)
