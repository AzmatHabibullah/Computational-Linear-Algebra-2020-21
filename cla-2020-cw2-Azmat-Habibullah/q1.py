import numpy as np
np.set_printoptions(linewidth = 350)


def construct_tridiag(c, d, m):
    """
    Construct an mxm tridiagonal matrix with diagonal c and super/subdiagonal b

    @param c: diagonal entries
    @param d: super/subdiagonal entries
    @param m: number of rows/columns of matrix

    @return: an mxm tridiagonal matrix with diagonal c and super/subdiagonal b
    """
    return np.diag([c for i in range(m)]) + np.diag([d for i in range(m - 1)], 1) + np.diag([d for i in range(m - 1)], -1)


def solve_tridiag_LU_inplace(c, d, B, returnAll=False):
    """
    Solve the tridiagonal system Axi = bi, with A with form in the question,
    using LU factorisation, with for i = 1 to n
    @param c: diagonal entries of m x m matrix A
    @param d: super/subdiagonal entries of A
    @param B: m x n matrix of containing the bi as columns
    @param returnAll: obvious
    @return: the m x n solution x to Ax = B
    """
    # initialise variables
    m, n = B.shape
    u = [c] * m                        # the vector u is the diagonal of U
    l = np.zeros(m, dtype=complex)     # the vector l is the subdiagonal of L
    Y = np.zeros((m, n), dtype=complex)
    X = np.zeros((m, n), dtype=complex)
    Y[0, :] = B[0, :]

    # one iteration forwards through vector entries
    for k in range(m - 1):
        l[k+1] = d/u[k]
        Y[k+1] = B[k+1] - l[k+1] * Y[k]
        u[k+1] -= l[k+1]*d

    X = np.zeros((m, n), dtype=complex)

    # one iteration backwards to back substitute and solve
    X[-1, :] = Y[-1] / u[-1]
    for i in range(m - 2, -1, -1):
        X[i, :] = (Y[i] - d * X[i + 1, :]) / u[i]


    if returnAll:
        return X, Y, l, u
    return X