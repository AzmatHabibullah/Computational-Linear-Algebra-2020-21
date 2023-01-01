"""
This file contains functions relating to the singular value decomposition (SVD) and truncation in Section 4
"""

import numpy as np
import matplotlib.pyplot as plt


def regularise(A, epsilon):
    """
    Regularise the Gram matrix by truncating singular values lower than epsilon
    :param A: Gram matrix
    :param epsilon: tolerance
    :return: lower-rank approximation to A
    """
    u, s, vh = np.linalg.svd(A)
    s[np.where(np.abs(s) < epsilon)] = 0
    true_s = np.zeros((u.shape[1], vh.shape[0]))
    true_s[:s.size, :s.size] = np.diag(s)
    return u.dot(true_s).dot(vh)


def gram_svals(n):
    """
    Compute the singular values of the n x n Gram matrix G
    :param n: dimensions of matrix m
    :return: singular values (== eigenvalues) of G
    """
    A = make_gram(n)
    u, v, sh = np.linalg.svd(A)
    return v


def make_gram(n):
    """
    Compute the n x n Gram matrix G for our Fourier extension example
    :param n: dimensions of G
    :return: G
    """
    A = 0.5*np.eye(n, dtype=complex)
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i == j:
                A[i-1, j-1] = 0.5
            else:
                A[i - 1, j - 1] = 1 / (np.pi * (j - i)) * np.sin((i - j) * np.pi/2)
    return A


