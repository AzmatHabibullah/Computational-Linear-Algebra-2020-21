import numpy as np


def randomQ(m):
    """
    Produce a random orthogonal mxm matrix.

    :param m: the matrix dimension parameter.
    
    :return Q: the mxm numpy array containing the orthogonal matrix.
    """
    Q, R = np.linalg.qr(np.random.randn(m, m))
    return Q


def randomR(m):
    """
    Produce a random upper triangular mxm matrix.

    :param m: the matrix dimension parameter.
    
    :return R: the mxm numpy array containing the upper triangular matrix.
    """
    
    A = np.random.randn(m, m)
    return np.triu(A)


def backward_stability_householder(m):
    """
    Verify backward stability for QR factorisation using Householder for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        Q1 = randomQ(m)
        R1 = randomR(m)
        A = Q1.dot(R1)
        Q2, R2 = np.linalg.qr(A)
        print("k =", k)
        print("Q difference:", np.linalg.norm(Q2 - Q1))
        print("R difference:", np.linalg.norm(R2 - R1))
        print("A difference:", np.linalg.norm(A - Q2.dot(R2)))

        # all errors are small

def solve_R(R, b):
    """
    Solve the system Rx=b where R is an mxm upper triangular matrix 
    and b is an m dimensional vector.

    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :param x: an m-dimensional numpy array
    """
                     
    m = R.shape[0]
    x = np.zeros(m)
    x[-1] = b[-1]/R[-1, -1]
    for i in range(m-2, -1, -1):
        x[i] = (b[i] - R[i, i:].dot(x[i:]))/R[i, i]
    return x


def back_stab_solve_R(m):
    """
    Verify backward stability for back substitution for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        A = np.random.randn(m, m)
        R = np.triu(A)
        b = np.random.randn(m)
        x = solve_R(R, b)
        # there exists perturbed input x + dx with ||dx||/||x|| = O(epsilon): R~. b~ exact solns.
        # then ||
        print("k =", k)
        print("difference:", np.linalg.norm(R.dot(x) - b))


def back_stab_householder_solve(m):
    """
    Verify backward stability for the householder algorithm
    for solving Ax=b for an m dimensional square system.

    :param m: the matrix dimension parameter.
    """
    raise NotImplementedError
