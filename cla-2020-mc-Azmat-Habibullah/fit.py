import numpy as np
from cla_utils import householder_ls
from svd import gram_svals, regularise


def polyfit(x, f, n, truncation=0, fourier=False):
    """
    Fit polynomial with Vandermonde method
    :param x: interpolation points
    :param f: data points
    :param n: interpolant/least squares degree
    :param fourier: if true, return fourier series coefficients
    :param truncation: if non-zero, truncate this many singular values
    :return: c, coefficient vector
    """
    A = np.vander(x, n + 1, increasing=True)
    if fourier:
        # check m >> 2n + 1
        B = np.concatenate([np.real(A), np.imag(A[:, 1:])], axis=1)
        if truncation > 0:
            if truncation > n+1:
                truncation = n
            epsilon = gram_svals(n)[n - truncation]
            B = regularise(B, epsilon)
        c = np.linalg.lstsq(B, f, rcond=None)[0]
        c = c[:n + 1] - 1.0j * np.concatenate([np.array([0]), c[n + 1:2 * n + 1]])
        return c
    m, n = A.shape
    if m > n: # not underdetermined
        return householder_ls(A, f)
    return np.linalg.lstsq(A, f, rcond=0)[0]


def polyval(c, s, fourier=False):
    """
    Evaluate polynomial from its coefficient vector c at points s
    :param c: coefficient vector
    :param s: evaluation points
    :param fourier: if true, use Fourier approximation
    :return: vector of evaluated points
    """
    n = c.size - 1
    B = np.vander(s, n+1, increasing=True)
    if fourier:
        return np.real(B.dot(c))
    return B.dot(c)


def polyfitA(X, f, n, fourier=False, truncation=0):
    """
    Fit polynomial with Arnoldi method
    :param x: interpolation points
    :param f: data points
    :param n: interpolant/least squares degree
    :param fourier: if true, return fourier series coefficients
    :param truncation: if non-zero, truncate this many singular values
    :return: Q, H, d: Q, H and d in construction
    """
    m, _ = X.shape
    H = np.zeros((n + 1, n), dtype=complex)
    Q = np.zeros((m, n + 1), dtype=complex)
    Q[:, 0] = np.ones(m)
    for k in range(n):
        v = X.dot(Q[:, k])
        H[:k + 1, k] = Q[:, :k + 1].conj().T.dot(v/m)
        v -= Q[:, :k + 1].dot(H[:k + 1, k])
        norm = np.linalg.norm(v)
        H[k + 1, k] = norm/np.sqrt(m)
        Q[:, k + 1] = v / H[k + 1, k]
    if fourier:
        B = np.concatenate([np.real(Q), np.imag(Q[:, 1:])], axis=1)
        if truncation > 0:
            if truncation > n+1:
                truncation = n
            epsilon = gram_svals(n)[n - truncation]
            B = regularise(B, epsilon)
        d = np.linalg.lstsq(B, f, rcond=None)[0]
        d = d[:n + 1] - 1.0j * np.concatenate([np.array([0]), d[n + 1:2 * n + 1]])
        return Q, H, d
    d = householder_ls(Q, f)
    return Q, H, d


def polyvalA(d, H, s, fourier=False):
    """
    Evaluate polynomial from its coefficient vector d and Hessenberg matrix at points s using Arnoldi
    :param d: coefficient vector
    :param H: Hessenberg matrix
    :param s: evaluation points
    :param fourier: if true, use Fourier approximation
    :return: vector of evaluated points
    """
    M = s.size
    _, n = H.shape
    W = np.zeros((M, n+1), dtype=complex)
    W[:, 0] = np.ones(M)
    for k in range(n):
        w = np.multiply(W[:, k], s)
        w -= np.dot(W[:, :k+1], H[:k+1, k])
        W[:, k + 1] = w / H[k + 1, k]
    y = np.dot(W, d)
    if fourier:
        return np.real(y)
    return y


def compute_interpolation(x, f, n, s, arnoldi=False, fourier=False, truncation=0):
    """
    General function to compute interpolation values
    :param x: interpolation data values
    :param f: function values
    :param n: degree of polynomial
    :param s: evaluation points
    :param arnoldi: if true, use Arnoldi method
    :param fourier: if true, approximate using Fourier series
    :param truncation: if non-zero, truncate this many singular values
    :return: vector of evaluated points
    """
    if fourier:
        z = np.e ** (1j * np.pi * x / 2)
        if arnoldi:
            Q, H, d = polyfitA(np.diag(z), f, n, fourier=fourier, truncation=truncation)
            return polyvalA(d, H, np.e ** (1j * np.pi * s / 2), fourier=fourier)
        else:
            c = polyfit(z, f, n, fourier=fourier, truncation=truncation)
            return polyval(c, np.e ** (1j * np.pi * s / 2), fourier=fourier)
    else:
        if arnoldi:
            Q, H, d = polyfitA(np.diag(x), f, n, fourier=fourier, truncation=truncation)
            return polyvalA(d, H, s, fourier=fourier)
        else:
            c = polyfit(x, f, n, fourier=fourier, truncation=truncation)
            return polyval(c, s, fourier=fourier)



