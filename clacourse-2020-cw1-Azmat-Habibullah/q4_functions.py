from numpy import *
import matplotlib.pyplot as plt
from cla_utils import *

dataset = loadtxt('pressure.dat')


# construct B for part a
def construct_B_part_a(p):
    B = np.zeros((2, 2 * p + 2))

    # first row of B is (p+1) 1s and (p+1) -1s
    B[0, :p+1] = 1
    B[0, p+1:] = -1

    # second row of B is 0 ... p 0 -1 ... -p
    B[1, :p+1] = np.linspace(0, p, p+1)
    B[1, p+1:] = -np.linspace(0, p, p+1)
    return B

# computes and returns various matrices related to A
def compute_A1_A2_AQ_A(Q, p):
    # compute blocks of A
    blockA1 = np.vander(dataset[:, 0][:51], p + 1, increasing=True)
    blockA2 = np.vander(dataset[:, 0][50:], p + 1, increasing=True)

    # construct A, AQ, A_1 and A_2
    A = np.block([
        [blockA1, np.zeros((51, p + 1))],
        [np.zeros((50, p + 1)), blockA2]
    ])
    AQ = A.dot(Q)
    A_1 = AQ[:, :2]
    A_2 = AQ[:, 2:]

    return A_1, A_2, AQ, A

# solve for y given y_1 and y_2
def solve_for_y(A1, A2, b, C, d):
    y1 = solve_R(C[:, :2], d)
    y2 = householder_ls(A2, b - A1.dot(y1))
    y = np.append(y1, y2)

    return y


def plot_part_c(precision, x1, x2, p):
    # plot the polynomials as well as the data
    plt.plot(dataset[:, 0], dataset[:, 1])
    x_vals_1 = arange(0, 1, 1 / precision)
    x_vals_2 = arange(1, 1.98, 1 / precision)
    plt.plot(x_vals_1, np.vander(x_vals_1, p + 1, increasing=True).dot(x1), label="Piece one")
    plt.plot(x_vals_2, np.vander(x_vals_2, p + 1, increasing=True).dot(x2), label="Piece two")
    plt.legend()
    plt.grid()
    plt.axvline()
    plt.axhline()
    title = "p = " + str(p)
    plt.title(title)
    plt.show()


def compute_thetas_per_block(thetas, interval_points):
    # compute the number of thetas in a given block; necessary to construct B
    # this means that the implementation does not require theta values to be eg linearly spaced
    return np.histogram(thetas, bins=interval_points)[0]


def construct_vandermonde_matrices(thetas, interval_points, p, M, N):
    # construct various Vandermonde matrices required to compute A and B
    theta_vander = np.vander(thetas, p + 1, increasing=True)
    theta_deriv_vander = np.zeros((N, p + 1))
    interval_vander = np.vander(interval_points, p + 1, increasing=True)
    interval_deriv_vander = np.zeros((M + 1, p + 1))

    # construct derivative Vandermonde matrices
    for i in range(p + 1):
        theta_deriv_vander[:, i] = i * theta_vander[:, i - 1]
        interval_deriv_vander[:, i] = i * interval_vander[:, i - 1]

    return theta_vander, theta_deriv_vander, interval_vander, interval_deriv_vander


def construct_A_part_d(p, M, N, thetas_per_block, theta_vander):
    A = np.zeros((N, M * (p + 1)))
    counter = 0
    # construct A block by block - each iteration places one block
    for i in range(thetas_per_block.size):
        A[counter:counter + thetas_per_block[i], i * (p + 1):(i + 1) * (p + 1)] = \
            theta_vander[counter:counter + thetas_per_block[i], :]
        counter += thetas_per_block[i]
    return A


# construct B iteratively for part d
def construct_B_part_d(p, M, interval_vander, interval_deriv_vander):
    B = np.zeros((2 * M, M * (p + 1)))
    for i in range(M-1):
        # create first row of the pair - continuity constraint
        B[2 * i, i * (p + 1):(i + 1) * (p + 1)] = interval_vander[i+1, :]
        B[2 * i, (i + 1) * (p + 1):(i + 2) * (p + 1)] = -interval_vander[i + 1, :]

        # create second row of the pair - continuous derivative constraint
        B[2 * i + 1, i * (p + 1):(i + 1) * (p + 1)] = interval_deriv_vander[i+1, :]
        B[2 * i + 1, (i + 1) * (p + 1):(i + 2) * (p + 1)] = -interval_deriv_vander[i + 1, :]
    if M != 1:
        # edge handling wrapping mechanism
        B[2 * (M - 1), (M - 1) * (p + 1):M * (p + 1)] = interval_vander[M, :]
        B[2 * (M - 1) + 1, (M - 1) * (p + 1):M * (p + 1)] = interval_deriv_vander[M, :]
        B[2 * (M - 1), :p + 1] = -interval_vander[0, :]
        B[2 * (M - 1) + 1, :p + 1] = -interval_deriv_vander[0, :]
    else:
        # special case M = 1
        B[0, :] = interval_vander[1, :]
        B[1, :] = interval_deriv_vander[1, :]
    return B


# following the method in part a, solve the least squares problems for individual components
def construct_y1_y2(A, Q, M, x_components_of_r, y_components_of_r):
    AQ = A.dot(Q)
    A1 = AQ[:, :2*M]
    A2 = AQ[:, 2*M:]

    new_y2 = householder_ls(A2, x_components_of_r)
    new_y4 = householder_ls(A2, y_components_of_r)

    y1 = np.append(np.zeros(2 * M), new_y2)
    y2 = np.append(np.zeros(2 * M), new_y4)

    return y1, y2


# plot each polynomial piece by piece
def plot_part_d(x_, y_, M, p, N, precision):
    for i in range(M):
        theta_vals = linspace(2*pi*i/M, 2*pi*(i+1)/M, precision)
        x_components = np.vander(theta_vals, p + 1, increasing=True).dot(x_[i * (p + 1):(i + 1) * (p + 1)])
        y_components = np.vander(theta_vals, p + 1, increasing=True).dot(y_[i * (p + 1):(i + 1) * (p + 1)])
        lbl = "m = " + str(i)
        plt.plot(x_components, y_components, label=lbl)
    title = "M = " + str(M) + ", p = " + str(p) + ", N = " + str(N)
    plt.title(title)
    plt.grid()
    plt.show()
