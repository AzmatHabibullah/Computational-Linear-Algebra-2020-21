import numpy as np
from q1 import *
from q2 import solve_Ax_b
import matplotlib.pyplot as plt


def construct_V(alpha, N):
    """
    Construct V
    @param alpha: alpha
    @param N: N
    @return: V
    """
    V = np.zeros((N, N), dtype=complex)
    exps = np.e ** (np.linspace(0, 2 * (N-1), N) * 1.0j * np.pi / N)
    for j in range(N):
        V[j, :] = alpha ** (-j / N) * exps ** j
    return V


def construct_r(p0, q0, deltat, deltax):
    """
    Construct r
    @param p0: p0
    @param q0: q0
    @param deltat: deltat
    @param deltax: deltax
    @return: r
    """
    M = p0.size
    r = np.zeros(2 * M)
    r[:M] = p0 + deltat / 2 * q0
    r[M:] = q0 + deltat / (2 * deltax ** 2) * (np.roll(p0, -1) - 2 * p0 + np.roll(p0, 1))
    return r


def left_mult_VxI_matrix(b, M, N, alpha):
    """
    Perform left multiplication by (V \otimes I) using algorithm in part e
    @param b: b
    @param M: M
    @param N: N
    @param alpha: alpha
    @return: (V \otimes I)b
    """
    b = b.reshape(N, 2*M)
    b = np.fft.ifft(b, axis=0) * N
    b = b.reshape(2*M*N)
    for i in range(N):
        b[2*M*i:2*M*(i+1)] = alpha ** (-i/N) * b[2*M*i:2*M*(i+1)]
    return b


def left_mult_VinvxI(b, M, N, alpha):
    """
    Perform left multiplication by (V^{-1} \otimes I) using algorithm in part e
    @param b: b
    @param M: M
    @param N: N
    @param alpha: alpha
    @return: (V^{-1} \otimes I)b
    """
    solution = np.zeros(2*M*N, dtype=complex)
    for i in range(N):
        solution[2 * M * i:2 * M * (i + 1)] = alpha ** (i / N) * b[2 * M * i:2 * M * (i + 1)]
    for i in range(2*M):
        solution[i::2*M] = np.fft.fft(solution[i::2*M])/N
    return solution


def construct_D1(N, alpha):
    """
    Construct D_1
    @param N: N
    @param alpha: alpha
    @return: D_1
    """
    return np.array([1 - alpha**(1/N) * np.e**(2*i*(N-1)*np.pi*1.0j/N) for i in range(N)])


def construct_D2(N, alpha):
    """
    Construct D_2
    @param N: N
    @param alpha: alpha
    @return: D_2
    """
    return 0.5*np.array([1 + alpha**(1/N) * np.e**(2*i*(N-1)*np.pi*1.0j/N) for i in range(N)])


def construct_B(deltat, deltax, M, return_B21=True):
    """
    Construct B
    @param deltat: deltat
    @param deltax: deltax
    @param M: M
    @param return_B21: if true, return B_{21}
    @return: B and/or B_{21}
    """
    B12 = -deltat * np.eye(M)
    B21 = deltat / (deltax ** 2) * construct_tridiag(2, -1, M)
    B21[0, M - 1] = -deltat / (deltax ** 2)
    B21[M - 1, 0] = -deltat / (deltax ** 2)

    B = np.zeros((2 * M, 2 * M))
    B[:M, M:] = B12
    B[M:, :M] = B21
    if return_B21:
        return B, B21
    return B


def construct_R(M, N, r, alpha, B, U):
    """
    Construct R
    @param M: M
    @param N: N
    @param r: r
    @param alpha: alpha
    @param B: B
    @param U: U
    @return: R
    """
    R = np.zeros(2*M*N)
    R[:2*M] = r + alpha*np.dot(-np.eye(2*M) + B/2, U[-2*M:])
    return R


def compute_Uhat(M, N, Rhat, D1, D2, B21, deltat, deltax):
    """
    Compute Uhat using algorithm in e
    @param M: M
    @param N: N
    @param Rhat: Rhat
    @param D1: D_1
    @param D2: D_2
    @param B21: B_{21}
    @param deltat: deltat
    @param deltax: deltax
    @return: Uhat
    """
    Uhat = np.zeros(2*M*N, dtype=complex)
    for k in range(N):
        rk1 = Rhat[k*2*M:k*2*M + M]
        rk2 = Rhat[M + k*2*M:k*2*M + 2*M]
        b = D1[k]*rk2 - D2[k]*B21.dot(rk1)
        c = 2*D2[k]**2 * deltat**2/(deltax**2) + D1[k]**2
        d = -D2[k]**2 * deltat**2/(deltax**2)
        qk = solve_Ax_b(c, d, b)
        pk = (deltat * D2[k] * qk + rk1)/D1[k]
        Uhat[k*2*M:(k+1)*2*M] = np.concatenate([pk, qk])
    return Uhat


def solve_system(p0, N, alpha, U0=None, q0=None, deltat=0.01, deltax=0.01, timesteps_to_plot=None):
    """
    Solve the pde
    @param p0: p0
    @param N: N
    @param alpha: alpha
    @param U0: initial guess for U
    @param deltat: deltat
    @param deltax: deltax
    @param timesteps_to_plot: timesteps to plot
    @return: 2xMxN array U as in question
    """
    # initialise variables
    M = p0.size
    if q0 is None:
        q0 = np.zeros(M)
    if U0 is None:
        U0 = np.zeros(2*M*N)
    r = construct_r(p0, q0, deltat, deltax)
    B, B21 = construct_B(deltat, deltax, M, return_B21=True)
    R = construct_R(M, N, r, alpha, B, U0)

    D1 = construct_D1(N, alpha)
    D2 = construct_D2(N, alpha)

    # perform algorithm in part e
    Rhat = left_mult_VinvxI(R, M, N, alpha)
    Uhat = compute_Uhat(M, N, Rhat, D1, D2, B21, deltat, deltax)
    U = left_mult_VxI_matrix(Uhat, M, N, alpha)

    # check U is real
    assert np.linalg.norm(U - U.real) < 10e-10
    U = U.real

    # plot U
    if timesteps_to_plot is not None:
        for t in range(timesteps_to_plot.size):
            plt.plot(np.linspace(0, 1, M), U[t*2*M:t*2*M + M], label="t")
        legend = ["t = " + str(i) for i in timesteps_to_plot]
        plt.legend(legend)
        plt.title("1D wave equation discretisation")
        plt.show()
    return U


if __name__ == '__main__':
    p0 = np.sin(np.linspace(0, np.pi, 100))
    solve_system(p0, 15, 0.00001, timesteps_to_plot = np.array([1, 3, 5, 7, 9]))