import numpy as np
import numpy.random as random

from cla_utils import householder_qr


def get_A100():
    """
    Return A100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    return A


def get_B100():
    """
    Return B100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A[np.tril_indices(m, -2)] = 0
    return A


def get_C100():
    """
    Return C100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    return A


def get_D100():
    """
    Return D100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    A[np.tril_indices(m, -2)] = 0
    A[np.triu_indices(m, 2)] = 0
    return A


def get_A3():
    """
    Return A3 matrix for investigating power iteration.
    
    :return A3: a 3x3 numpy array.
    """

    return np.array([[ 0.68557183+0.46550108j,  0.12934765-0.1622676j ,
                    0.24409518+0.25335939j],
                  [ 0.1531015 +0.66678983j,  0.45112492+0.18206976j,
                    -0.02633966+0.43477693j],
                  [-0.10817164-1.16879196j, -0.18446849+0.03755672j,
                   0.06430325-0.44757084j]])


def get_B3():
    """
    Return B3 matrix for investigating power iteration.

    :return B3: a 3x3 numpy array.
    """
    return np.array([[ 0.46870499+0.37541453j,  0.19115959-0.39233203j,
                    0.12830659+0.12102382j],
                  [ 0.90249603-0.09446345j,  0.51584055+0.84326503j,
                    -0.02582305+0.23259079j],
                  [ 0.75419973-0.52470311j, -0.59173739+0.48075322j,
                    0.51545446-0.21867957j]])


def pow_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a matrix A, apply the power iteration algorithm with initial
    guess x0, until either

    ||r|| < tol where

    r = Ax - lambda*x,

    or the number of iterations exceeds maxit.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of power iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing all \
    the iterates.
    :return lambda0: the final eigenvalue.
    """
    if store_iterations:
        m, _ = A.shape
        v = np.zeros((m, maxit))
        v[:, 0] = x0
    lambda_k = 0
    v_k_minus_1 = x0
    v_k = np.zeros(maxit)

    for k in range(1, maxit):
        w = A.dot(v_k_minus_1)
        v_k = w / np.linalg.norm(w)
        lambda_k = v_k.T.conj().dot(A.dot(v_k))
        r = A.dot(v_k) - lambda_k * v_k
        if store_iterations:
            v[:, k] = v_k
        v_k_minus_1 = v_k
        if np.linalg.norm(r) < tol:
            if store_iterations:
                return v, lambda_k
            return v_k, lambda_k
    if store_iterations:
        return v, lambda_k
    return v_k, lambda_k


def inverse_it(A, x0, mu, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the inverse iteration algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param mu: a floating point number, the shift parameter
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """
    m, _ = A.shape
    if store_iterations:
        v = np.zeros((m, maxit))
        lambdas = np.zeros(m)
        v[:, 0] = x0
    lambda_k = 0
    v_k_minus_1 = x0
    v_k = np.zeros(maxit)

    for k in range(1, maxit):
        w = np.linalg.solve(A - mu*np.eye(m), v_k_minus_1)
        v_k = w / np.linalg.norm(w)
        lambda_k = v_k.T.conj().dot(A.dot(v_k))
        r = A.dot(v_k) - lambda_k * v_k
        if store_iterations:
            v[:, k] = v_k
            lambdas[k] = lambda_k
        v_k_minus_1 = v_k
        if np.linalg.norm(r) < tol:
            if store_iterations:
                return v, lambdas
            return v_k, lambda_k
    if store_iterations:
        return v, lambdas
    return v_k, lambda_k


def rq_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the Rayleigh quotient algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """

    m, _ = A.shape
    if store_iterations:
        v = np.zeros((m, maxit))
        lambdas = np.zeros(m)
        v[:, 0] = x0
    v_k_minus_1 = x0
    lambda_k_minus_1 = v_k_minus_1.T.conj().dot(A.dot(v_k_minus_1))
    v_k = np.zeros(maxit)

    for k in range(1, maxit):
        w = np.linalg.solve(A - lambda_k_minus_1*np.eye(m), v_k_minus_1)
        v_k = w / np.linalg.norm(w)
        lambda_k = v_k.T.conj().dot(A.dot(v_k))
        r = A.dot(v_k) - lambda_k * v_k
        if store_iterations:
            v[:, k] = v_k
            lambdas[k] = lambda_k
        v_k_minus_1 = v_k
        if np.linalg.norm(r) < tol:
            if store_iterations:
                return v, lambdas
            return v_k, lambda_k
    if store_iterations:
        return v, lambdas
    return v_k, lambda_k


def pure_QR(A, maxit=10000, tol=10**-12, return_t_errors=False):
    """
    For matrix A, apply the QR algorithm and return the result.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance

    :return Ak: the result
    """
    m, _ = A.shape
    Ak = A
    t_errors = np.zeros(maxit)
    for k in range(0, maxit):
        Q, R = householder_qr(Ak)
        Ak = R.dot(Q)
        norm = np.linalg.norm(Ak.diagonal(-1))
        if return_t_errors:
            t_errors[k] = np.linalg.norm(Ak[m-1, m-2])
        if norm < tol:
            break
    if return_t_errors:
        return Ak, t_errors[:k+1]
    return Ak
