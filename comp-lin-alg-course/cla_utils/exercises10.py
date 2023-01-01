from cla_utils import *


def arnoldi(A, b, k):
    """
    For a matrix A, apply k iterations of the Arnoldi algorithm,
    using b as the first basis vector.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array, the starting vector
    :param k: integer, the number of iterations

    :return Q: an mxk dimensional numpy array containing the orthonormal basis
    :return H: a (k+1)xk dimensional numpy array containing the upper \
    Hessenberg matrix
    """
    m, _ = A.shape
    H = np.zeros((k + 1, k), dtype=complex)
    Q = np.zeros((m, k + 1), dtype=complex)
    Q[:, 0] = b / np.linalg.norm(b)
    for n in range(k):
        v = A.dot(Q[:, n])
        H[:n + 1, n] = Q[:, :n + 1].conj().T.dot(v)
        v -= Q[:, :n+1].dot(H[:n+1, n])
        norm = np.linalg.norm(v)
        H[n + 1, n] = norm
        Q[:, n + 1] = v/norm

    return Q, H

def GMRES(A, b, apply_pc = None, maxit=10000, tol=10e-9, x0=None, return_residual_norms=False,
          return_residuals=False):
    """
    For a matrix A, solve Ax=b using the basic GMRES algorithm.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param x0: the initial guess (if not present, use b)
    :param return_residual_norms: logical
    :param return_residuals: logical

    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of \
    the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual \
    at iteration k
    """
    if x0 is None:
        x0 = b
    m, _ = A.shape
    H = np.zeros((maxit + 1, maxit), dtype=complex)
    Q = np.zeros((m, maxit + 1), dtype=complex)
    X = np.zeros((m, maxit + 1), dtype=complex)
    if apply_pc is not None:
        b = apply_pc(b)
    residuals = np.zeros(maxit)
    residual_norms = np.zeros(maxit)
    Q[:, 0] = b / np.linalg.norm(b)
    qn = b/np.linalg.norm(b)
    e1 = np.zeros(maxit)
    e1[0] = 1
    for n in range(maxit):
        if apply_pc is not None:
            v = apply_pc(A.dot(Q[:, n]))
        else:
            v = A.dot(Q[:, n])
        H[:n + 1, n] = Q[:, :n + 1].conj().T.dot(v)
        v -= Q[:, :n + 1].dot(H[:n + 1, n])
        norm = np.linalg.norm(v)
        H[n + 1, n] = norm
        Q[:, n + 1] = v / norm
        y = householder_ls(H[:n+2, :n+1], np.linalg.norm(b) * e1[:n+2])
        X[:, n] = Q[:, :n+1].dot(y)
        residual = H[:n+2, :n+1].dot(y) - np.linalg.norm(b)*e1[:n+2]
        residual_norm = np.linalg.norm(residual)
        if return_residuals:
            residuals[n] = residual
        if return_residual_norms:
            residual_norms[n] = residual_norm
        if residual_norm < tol:
            if return_residuals and return_residual_norms:
                return X[:, n], n, residuals, residual_norms
            if return_residuals:
                return X[:, n], n, residuals
            if return_residual_norms:
                return X[:, n], n, residual_norms[:n+1]
            return X[:, n], n
    if return_residuals and return_residual_norms:
        return X[:, n], -1, residuals, residual_norms
    if return_residuals:
        return X[:, n], -1, residuals
    if return_residual_norms:
        return X[:, n], -1, residual_norms # todo tidy return up
    return X[:, n], -1 # todo fix return_residual

    """    if x0 is None:
            x0 = b
        m, _ = A.shape
        H = np.zeros((maxit + 1, maxit), dtype=complex)
        Q = np.zeros((m, maxit + 1), dtype=complex)
        X = np.zeros((m, maxit + 1), dtype=complex)
        Q[:, 0] = b / np.linalg.norm(b)
        qn = b/np.linalg.norm(b)
        e1 = np.zeros(maxit)
        e1[0] = 1
        hq = np.zeros((maxit + 1, maxit + 1))
        hr = np.zeros((maxit + 1, maxit))
        for n in range(maxit):
            v = A.dot(Q[:, n])
            H[:n + 1, n] = Q[:, :n + 1].conj().T.dot(v)
            v -= Q[:, :n + 1].dot(H[:n + 1, n])
            norm = np.linalg.norm(v)
            H[n + 1, n] = norm
            Q[:, n + 1] = v / norm
            # househodler_ls
            Hb = np.concatenate([H[:n+2, :n+1], np.array([np.linalg.norm(b) * e1[:n+2]]).conj().T], axis=1)
            hq, hr = householder_qr(H[:n+2, :n+1])
            #hr[n+1, n+1] = -hr[n+1, n+1]
            RQstarb = householder(Hb, n+1)
            y = sp.solve_triangular(hr[:n + 1, :n + 1], hq.dot(np.linalg.norm(b) * e1[:n + 2])[:n + 1])
            #y = sp.solve_triangular(RQstarb[:n+1, :n+1], RQstarb[:n+1, -1])
            #end
            X[:, n] = Q[:, :n+1].dot(y)
            if np.linalg.norm(H[:n+2, :n+1].dot(y) - np.linalg.norm(b)*e1[:n+2]) < tol:
                return X[:, n], n

        return X[:, n], -1"""

def get_AA100():
    """
    Get the AA100 matrix.

    :return A: a 100x100 numpy array used in exercises 10.
    """
    AA100 = np.fromfile('cla_utils\\AA100.dat', sep=' ')
    AA100 = AA100.reshape((100, 100))
    return AA100


def get_BB100():
    """
    Get the BB100 matrix.

    :return B: a 100x100 numpy array used in exercises 10.
    """
    BB100 = np.fromfile('cla_utils\\BB100.dat', sep=' ')
    BB100 = BB100.reshape((100, 100))
    return BB100


def get_CC100():
    """
    Get the CC100 matrix.

    :return C: a 100x100 numpy array used in exercises 10.
    """
    CC100 = np.fromfile('cla_utils\\CC100.dat', sep=' ')
    CC100 = CC100.reshape((100, 100))
    return CC100
