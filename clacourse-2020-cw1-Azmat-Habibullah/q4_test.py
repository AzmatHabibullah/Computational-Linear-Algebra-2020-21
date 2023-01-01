import pytest
from q4 import *
from q4_functions import *
from numpy import *


@pytest.mark.parametrize('z', [5, 10, 12, 13])
def test_part_c(z):
    random.seed(1241)
    p = random.randint(1, 10) * z
    B = construct_B_part_a(p)

    Q, R = householder_qr(B.T)

    # check QR factorises well
    assert(np.max(np.abs(Q.dot(R) - B.T)) < 10**-10)

    C = R.T  # lower triangular

    A_1, A_2, AQ, A = compute_A1_A2_AQ_A(Q, p)

    # check A non zero
    assert(np.max(np.abs(A)) > 0)

    y = solve_for_y(A_1, A_2, b, C, d)
    x = Q.dot(y)

    # check first entry of y is 0, as expected theoretically
    assert(y[0] < 10**-10)

    # check x non zero
    assert(np.max(np.abs(x)) > 0)


@pytest.mark.parametrize('z', [1, 2, 3, 4])
def test_part_d(z):
    random.seed(1241)
    p = random.randint(1, 20) * z
    M = random.randint(1, 40) * z
    N = random.randint(200, 500) * z

    thetas = linspace(0, 2 * pi, N)
    r = 2 - 2 * sin(-3 * thetas)
    x_components_of_r = r * np.cos(thetas)
    y_components_of_r = r * np.sin(thetas)

    interval_points = linspace(0, 2 * pi, M + 1)

    thetas_per_block = compute_thetas_per_block(thetas, interval_points)
    print(thetas_per_block)

    theta_vander, theta_deriv_vander, interval_vander, interval_deriv_vander = \
        construct_vandermonde_matrices(thetas, interval_points, p, M, N)

    A = construct_A_part_d(p, M, N, thetas_per_block, theta_vander)
    B = construct_B_part_d(p, M, interval_vander, interval_deriv_vander)

    # check A not empty
    assert(np.max(np.abs(A)) > 10**-10)

    # check B not empty
    assert(np.max(np.abs(B)) > 10**-10)

    # now follow a similar method as above
    Q, R = householder_qr(B.T)

    # construct y1 and y2
    y1, y2 = construct_y1_y2(A, Q, M, x_components_of_r, y_components_of_r)

    # find x and y polynomial coefficients
    x_polynomial_coefficients = Q.dot(y1)
    y_polynomial_coefficients = Q.dot(y2)

    # check non empty
    assert(np.max(np.abs(x_polynomial_coefficients)) > 10**-10)
    assert(np.max(np.abs(y_polynomial_coefficients)) > 10**-10)