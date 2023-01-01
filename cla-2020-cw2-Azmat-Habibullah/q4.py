import cla_utils.exercises10 as e10
from cla_utils import householder_ls
from cla_utils import GMRES
from cla_utils import solve_U
from scipy.sparse import csgraph
import matplotlib.pyplot as plt
import numpy as np


def make_upper_triangular_pc(M):
    """
    Upper triangular preconditioner
    @param M: upper triangular preconditioning matrix
    @return: function to solve Mx = b for upper triangular M
    """
    def apply_pc(b):
        return solve_U(M, b, one_dim=True)
    return apply_pc


def best_c(A, return_errors=False):
    """
    For given matrix A, determine the c that minimises the error |I - M^{-1}A|
    with M = cU, where U is the upper triangular part of A
    @param A: matrix to determine c for
    @param return_errors: 2x991 array: if true, return all of the errors for various c.
                          First row is values of c, second is the error
    @return: the value of c in [0.1, 10] which minimises the error. Most errors were
             experimenally determined to fall within this range hence it has been choosed
             for efficiency reasons.
    """
    m, _ = A.shape
    errors = np.zeros((2, 991))
    cs = np.linspace(0.1, 10, 991)
    errors[0, :] = cs
    for i in range(991):
        M = cs[i] * np.triu(A)
        y = np.linalg.norm(np.eye(m) - np.linalg.inv(M).dot(A))
        errors[1, i] = y
    best_c = errors[0, np.argmin(errors[1, :])]
    if return_errors:
        return best_c, errors
    return best_c


def plot(pc_iterates, pc_norms, non_pc_iterates, non_pc_norms):
    """
    Plot the graph comparing the residual norms with and without preconditioning
    @param pc_iterates: preconditioning number of iterates
    @param pc_norms: preconditioned norms
    @param non_pc_iterates: non preconditioned number of iterates
    @param non_pc_norms: non preconditioned norms
    """
    plt.plot(np.linspace(1, pc_iterates+1, pc_iterates+1), pc_norms, label="Preconditioned")
    plt.plot(np.linspace(1, non_pc_iterates+1, non_pc_iterates+1), non_pc_norms, label="Non-preconditioned")
    plt.legend()
    plt.title("Conditioning vs preconditioning")
    plt.yscale("log")
    plt.xlabel("Iterate")
    plt.ylabel("Residual norm")
    plt.show()


def run_investigation(B):
    """
    Run the investigation for a given matrix adjacency graph B.
    Investigation runs with A = I + L, where L is the graph Laplacian of B
    @param B: Adjacency matrix B to form L from.
    """
    m, _ = B.shape
    I = np.eye(m)
    L = csgraph.laplacian(B, normed=False)
    A = I + L
    b = np.random.rand(m)
    c = best_c(A)

    # compute residuals
    print("Best c:", c)
    M = c * np.triu(A)
    apply_pc = make_upper_triangular_pc(M)

    _, pc_iterates, pc_norms = GMRES(A, b, apply_pc=apply_pc, return_residual_norms=True)
    MinvA_eigs = np.linalg.eig(np.linalg.inv(M).dot(A))[0]
    A_eigs = np.linalg.eig(A)[0]
    y = np.linalg.norm(I - np.linalg.inv(M).dot(A))
    assert (np.abs(1 - MinvA_eigs) <= y).all(), \
        "No c found satisfying |I - M^{-1}A| = c < 1. Increase the range or use a different matrix"

    _, non_pc_iterates, non_pc_norms = GMRES(A, b, return_residual_norms=True)

    plot_eigenvalues(A_eigs, "A eigenvalues")
    plot_eigenvalues(MinvA_eigs, "M^{-1}A eigenvalues")
    plot(pc_iterates, pc_norms, non_pc_iterates, non_pc_norms)

    # check decreasing sequences and print residuals
    assert all(earlier >= later for earlier, later in zip(pc_norms, pc_norms[1:]))
    print("Preconditioned residual norms:", pc_norms)

    assert all(earlier >= later for earlier, later in zip(non_pc_norms, non_pc_norms[1:]))
    print("Non-preconditioned residual norms:", non_pc_norms)


def plot_eigenvalues(eigs, title):
    plt.plot(np.real(eigs), np.imag(eigs), 'x')
    plt.title(title)
    plt.xlabel("Real part")
    plt.ylabel("Imaginary part")
    plt.show()


if __name__ == '__main__':
    # part d
    # --- random graph
    B = np.random.rand(6, 6)
    run_investigation(B)

    # --- random graph with integer weights
    B = np.random.randint(0, 100, (100, 100))
    run_investigation(B)




