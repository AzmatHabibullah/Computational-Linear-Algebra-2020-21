import numpy as np


def operator_2_norm(A):
    """
    Given a real mxn matrix A, return the operator 2-norm.

    :param A: an mxn-dimensional numpy array

    :return o2norm: the norm
    """
    # operator 2-norm is found using Lagrange multipliers to be the max e-val of (A^T)A
    Y = A.T.dot(A)
    evals = np.linalg.eig(Y)
    max_eval = np.amax((evals)[0])
    return np.sqrt(max_eval)


def cond(A):
    """
    Given a real mxn matrix A, return the condition number in the 2-norm.

    :return A: an mxn-dimensional numpy array

    :param ncond: the condition number
    """
    # condition number is found to be the op 2-norm of A * op 2-norm of A^(-1)
    # but the op 2-norm of A^(-1) is just 1/sqrt(min eval of (A^T)A)
    # this is because (A^T)A and A(A^T) have the same eigenvales
    A_norm = operator_2_norm(A)
    Y = A.T.dot(A)
    evals = np.linalg.eig(Y)
    max_eval = np.amax((evals)[0])
    min_eval = np.amin((evals)[0])

    """
    (A^T)A is semi-positive definite so negative eigenvalues are
     due to floating point errors so we return unbounded if an eigenvalue is less than 0
    """
    if min_eval < 0:
        return "unbounded"

    A_inverse_norm = 1/np.sqrt(min_eval)
    ncond = A_norm * A_inverse_norm
    return ncond
