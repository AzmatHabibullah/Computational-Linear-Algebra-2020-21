import numpy as np
import timeit
import numpy.random as random

# pre-construct a matrix in the namespace to use in tests
random.seed(1651)
A0 = random.randn(500, 500)
x0 = random.randn(500)


def basic_matvec(A, x):
    """
    Elementary matrix-vector multiplication.

    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    returns an m-dimensional numpy array which is the product of A with x

    This should be implemented using a double loop over the entries of A

    :return b: m-dimensional numpy array
    """

    print("Implementing the standard definition b_i = sum_{j = 1}^n a_{ij} x_j")
    dims = A.shape
    b = np.zeros(dims[0])
    for i in range(dims[0]):
        for j in range(dims[1]):
            b[i] = b[i] + A[i][j]*x[j]
    return b

def column_matvec(A, x):
    """
    Matrix-vector multiplication using the representation of the product
    Ax as linear combinations of the columns of A, using the entries in
    x as coefficients.


    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    :return b: an m-dimensional numpy array which is the product of A with x

    This should be implemented using a single loop over the entries of x
    """

    dims = A.shape
    b = np.zeros(dims[0])
    for i in range(dims[1]):
        b = b + A[:, i] * x[i]

    return b


def timeable_basic_matvec():
    """
    Doing a matvec example with the basic_matvec that we can
    pass to timeit.
    """

    b = basic_matvec(A0, x0) # noqa


def timeable_column_matvec():
    """
    Doing a matvec example with the column_matvec that we can
    pass to timeit.
    """

    b = column_matvec(A0, x0) # noqa


def timeable_numpy_matvec():
    """
    Doing a matvec example with the builtin numpy matvec so that
    we can pass to timeit.
    """

    b = A0.dot(x0) # noqa


def time_matvecs():
    """
    Get some timings for matvecs.
    """

    print("Timing for basic_matvec")
    print(timeit.Timer(timeable_basic_matvec).timeit(number=1))
    print("Timing for column_matvec")
    print(timeit.Timer(timeable_column_matvec).timeit(number=1))
    print("Timing for numpy matvec")
    print(timeit.Timer(timeable_numpy_matvec).timeit(number=1))


def rank2(u1, u2, v1, v2):
    """
    Return the rank2 matrix A = u1*v2^* + u2*v2^*.

    :param u1: m-dimensional numpy array
    :param u1: m-dimensional numpy array
    :param v1: n-dimensional numpy array
    :param v2: n-dimensional numpy array
    """
    B = np.outer(u1, v1.conjugate())
    C = np.outer(u2, v2.conjugate())
    print(B)
    print(C)
    A = np.add(B, C)
    print(np.linalg.matrix_rank(A))
    return A


def rank1pert_inv(u, v):
    """
    Return the inverse of the matrix A = I + uv^*, where I
    is the mxm dimensional identity matrix, with

    :param u: m-dimensional numpy array
    :param v: m-dimensional numpy array
    """

    """
    Inverse is of the form A^-1 = I + alpha u v^*, and multiplying AA^-1 we
    need alpha + 1 + alpha v^* u = 0, so alpha = -1/(v^* u + 1), v^* u =/= -1
    """
    
    vConjugate = v.conjugate()
    alpha = -1/(vConjugate.dot(u) + 1)
    Ainv = np.eye(u.shape[0]) + alpha*np.outer(u, vConjugate)

    return Ainv


def ABiC(Ahat, xr, xi):
    """Return the real and imaginary parts of z = A*x, where A = B + iC
    with

    :param Ahat: an mxm-dimensional numpy array with Ahat[i,j] = B[i,j] \
    for i<=j and Ahat[i,j] = C[i,j] for i>j.

    :return zr: m-dimensional numpy arrays containing the real part of z.
    :return zi: m-dimensional numpy arrays containing the imaginary part of z.
    """

    m = Ahat.shape[0]
    B = np.eye(m)
    C = np.eye(m)
    for i in range(m):
        B[i][i] = Ahat[i][i] 
        C[i][i] = 0
        for j in range(m):
            if i > j:
                B[i][j] = Ahat[i][j]
                B[j][i] = B[i][j]
            if i < j:
                C[i][j] = Ahat[i][j]
                C[j][i] = -C[i][j]
    
    zr = np.zeros(m)
    zi = np.zeros(m)
    for i in range(m):
        zr = zr + B[:, i] * xr[i] - C[:, i] * xi[i]
        zi = zi + B[:, i] * xi[i] + C[:, i] * xr[i]
    A = B + 1j*C

    return zr, zi
