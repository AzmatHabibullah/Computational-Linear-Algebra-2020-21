from q4_functions import *
from cla_utils import householder_qr

dataset = loadtxt('pressure.dat')

# part a
b = dataset[:, 1][:51]
b = np.append(b, dataset[:, 1][50:])
d = [0, -5]

# plotting precision
precision = 250

if __name__ == "__main__":
    # upper bound for degrees to plot for part c
    upper_p = 10

    # part c
    error = np.zeros(upper_p)
    error[0] = 70.42040478350594  # for calculating accuracy increase from first value
    for p in range(1, upper_p):
        # construct B
        B = construct_B_part_a(p)

        # construct C
        Q, R = householder_qr(B.T)
        C = R.T

        # transform coordinates
        A_1, A_2, __, A = compute_A1_A2_AQ_A(Q, p)
        y = solve_for_y(A_1, A_2, b, C, d)
        x = Q.dot(y)

        # split the polynomials
        x1 = x[:p + 1]
        x2 = x[p + 1:]

        # plot the polynomials
        plot_part_c(precision, x1, x2, p)

        # output the errors
        error[p] = around(sum(np.abs(b - A.dot(np.append(x1, x2)))), 1)
        percentage = around((1 - error[p]/error[p-1]) * 100, 1)
        print("p = ", p, ": error = ", error[p], " - ", percentage, "% improvement", sep='')

    # part d variables
    p = 4
    M = 9
    N = 200

    # generate a random sample of theta values, a cardioid, and find Cartesian representation
    thetas = np.sort(np.random.sample(N)) * 2 * np.pi
    r = 2 - 2*sin(-3*thetas)
    x_components_of_r = r * np.cos(thetas)  # corresponds to c_x in the report
    y_components_of_r = r * np.sin(thetas)  # corresponds to c_y in the report

    # plot Cartesian coordinates
    plt.plot(x_components_of_r, y_components_of_r)

    # construct the intervals
    interval_points = linspace(0, 2 * pi, M + 1)

    # compute number of thetas per block - not constant since we randomly generated thetas
    thetas_per_block = compute_thetas_per_block(thetas, interval_points)

    # construct Vandermonde matrices for use in constructing A and B
    theta_vander, theta_deriv_vander, interval_vander, interval_deriv_vander = \
        construct_vandermonde_matrices(thetas, interval_points, p, M, N)

    # construct matrices A and B
    A = construct_A_part_d(p, M, N, thetas_per_block, theta_vander)
    B = construct_B_part_d(p, M, interval_vander, interval_deriv_vander)

    # now follow a similar method as above
    Q, R = householder_qr(B.T)

    # construct y1 and y2
    y1, y2 = construct_y1_y2(A, Q, M, x_components_of_r, y_components_of_r)

    # find x and y polynomial coefficients
    x_polynomial_coefficients = Q.dot(y1)
    y_polynomial_coefficients = Q.dot(y2)

    # plot part d
    plot_part_d(x_polynomial_coefficients, y_polynomial_coefficients, M, p, N, precision)
