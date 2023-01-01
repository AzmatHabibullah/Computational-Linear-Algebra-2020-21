from cla_utils import *
import matplotlib.pyplot as plt


def qr_factor_tri(A, return_vecs=False):
    """
    qr_factor_tri algorithm in part b. Apply 2x2 Householder reflectors in place to reduce a matrix to upper triangular
    @param A: matrix to reduce
    @param return_vecs: if true, return the reflectors
    @return: upper triangular R
    """
    # initialise variables
    m, n = A.shape
    e1 = np.array([1, 0])
    V = np.zeros((2, m-1))
    e1[0] = 1
    for k in range(n-1):
        # apply 2x2 Househodler reflectors
        x0 = A[k:k+2, k]
        if x0[0] == 0:
            sgn = 1
        else:
            sgn = np.sign(x0[0])
        v_k0 = sgn*np.linalg.norm(x0)*e1 + x0
        v_k0 = v_k0/np.linalg.norm(v_k0)
        V[:, k] = v_k0
        A[k:k+2, k:k+3] -= 2*np.outer(v_k0, v_k0.conjugate().T.dot(A[k:k+2, k:k+3]))
    if return_vecs:
        return V


def qr_alg_tri(A, shift=None, return_t_errors=False, max_iterations=10000, tolerance=10 ** -12):
    """
    modified qr_alg_tri with optional Wilkinson shift
    unshifted QR algorithm for tridiagonal matrices
    @param A:
    @param shift: if true, apply Wilkinson shift
    @param return_t_errors: if true, return |T_{m, m-1}|
    @param max_iterations: maximum number of iterations
    @param tolerance: tolerance
    @return: upper triangular R
    """
    Ak = 1.0*A  # make a copy to prevent creation of new A instances in later questions
    m, _ = A.shape
    # handle edge cases
    if shift is not None and (m==1 or np.linalg.norm(np.linalg.norm(A) - np.linalg.norm(np.diag(A))) < tolerance):
        if return_t_errors:
            if m != 1:
                return A, np.array([np.linalg.norm([A[m-1, m-2]])])
            return A, np.array([])
        return A
    # reduce to tridiagonal
    hessenberg(Ak)
    t_errors = np.zeros(max_iterations)
    for i in range(max_iterations):
        if shift is not None:
            # compute Wilkinson shift coefficients
            a = Ak[m - 1, m - 1]
            b = Ak[m - 1, m - 2]
            delta = (Ak[m - 2, m - 2] - a) / 2
            if delta == 0:
                sgn = 1
            else:
                sgn = np.sign(delta)
            mu = a - sgn * b ** 2 / (np.linalg.norm(delta) + np.sqrt(delta ** 2 + b ** 2))
            B = Ak - mu * np.eye(m)
            Vk = qr_factor_tri(B, return_vecs=True)
        else:
            Vk = qr_factor_tri(Ak, return_vecs=True)
            # apply Householder rotations using reflectors
        for k in range(m - 1):
            Qk = np.eye(m)
            vk = Vk[:, k]
            F = np.eye(m - k)
            F[:2, :2] -= 2 * np.outer(vk, vk.T) / np.linalg.norm(vk)
            Qk[k:, k:] = F.T
            if shift is not None:
                B = B.dot(Qk)
            else:
                Ak = Ak.dot(Qk)
        if shift is not None:
            # apply shift again
            Ak = B + mu*np.eye(m)
            # detaching
            for j in range(m - 1):
                if np.linalg.norm(Ak[j, j + 1]) < tolerance:
                    Ak[j, j + 1] = 0
                    Ak[j + 1, j] = 0
                    print("Detaching with")
                    print(Ak)
                    if return_t_errors:
                        Ak[:j + 1, :j + 1], its = qr_alg_tri(Ak[:j + 1, :j + 1], shift=True, return_t_errors=True)
                        Ak[j + 1:, j + 1:], its0 = qr_alg_tri(Ak[j + 1:, j + 1:], shift=True, return_t_errors=True)
                        t_errors = np.append(t_errors, its)
                        t_errors = np.append(t_errors, its0)
                    else:
                        Ak[:j + 1, :j + 1]= qr_alg_tri(Ak[:j + 1, :j + 1], shift=True)
                        Ak[j + 1:, j + 1:] = qr_alg_tri(Ak[j + 1:, j + 1:], shift=True)
        # check break conditions
        norm = np.linalg.norm(Ak[m-1, m-2])
        if return_t_errors:
            t_errors[i] = norm
        if norm < tolerance:
            break
    if return_t_errors:
        return Ak, t_errors[np.nonzero(t_errors)]
    return Ak


def concatenate(A, shift=None, compare_to_pure=True):
    """
    Concatenation for part e and f. Plot values of |T_{m, m-1}| at each iteration for the algorithms
    once complete, truncate the last row and column and repeat
    @param A: matrix to work with
    @param shift: if true, apply Wilkinson shift
    @param compare_to_pure: if true, plot pure qr values too
    @return: iterations in concatenation scheme
    """
    # initialise variables
    A0 = 1.0*A
    hessenberg(A)
    m, _ = A.shape
    # apply qr alg tri
    alg_T, alg_iterates = qr_alg_tri(A, shift=shift, return_t_errors=True)
    if compare_to_pure:
        pure_T, pure_iterates = pure_QR(A0, return_t_errors=True)
    for i in range(1, m - 1):
        # concatenate
        alg_T = alg_T[:-1, :-1]
        alg_T, alg_to_append = qr_alg_tri(alg_T, shift=shift, return_t_errors=True)
        alg_iterates = np.append(alg_iterates, alg_to_append)
        if compare_to_pure:
            pure_T = pure_T[:-1, :-1]
            pure_T, pure_to_append = pure_QR(pure_T, return_t_errors=True)
            pure_iterates = np.append(pure_iterates, pure_to_append)
    if compare_to_pure:
        return alg_iterates, pure_iterates
    return alg_iterates


def plot_iterates(alg_iterates, title, pure_iterates=None, alg_name="Unshifted qr_alg_tri"):
    """
    Plot iterates in part e, f
    @param alg_iterates: entries to plot
    @param title: title of plot
    @param pure_iterates: if provided, plot the pure qr iterates on the same axis
    @param alg_name: name of algorithm (unshifted or Wilkinson)
    """
    plt.plot(np.linspace(1, alg_iterates.size, alg_iterates.size), alg_iterates, label=alg_name)
    if pure_iterates is not None:
        plt.plot(np.linspace(1, pure_iterates.size, pure_iterates.size), pure_iterates, label="Pure QR")
    plt.legend()
    plt.yscale('log')
    plt.xlabel("Iterate")
    plt.ylabel("Concatenated errors")
    plt.title(title)
    plt.show()


def construct_A(m):
    """
    Construct A in part d
    @param m: dimensions of A
    @return: A
    """
    A = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            A[i, j] = 1/((i+1) + (j+1) + 1)
    return A


def construct_A2(m):
    """
    Construct A in part g
    @param m: matrix dimensions
    @return:
    """
    O = np.ones((m, m))
    D = np.diag(np.linspace(m, 1, m))
    return O + D


def plot_superdiag_norms(A, shift=None, alg_name="Unshifted qr_alg_tri"):
    """
    plot |T_{m, m-1}| for qr alg tri for given matrix, for part d
    @param A: matrix to work with
    @param shift: if true, apply Wilkinson shift
    @param alg_name: name of algorithm for plot (unshifted or Wilkinson)
    """
    _, alg_iterates = qr_alg_tri(A, shift=shift, return_t_errors=True)
    alg_x_vals = np.linspace(1, alg_iterates.size, alg_iterates.size)
    plt.plot(alg_x_vals, alg_iterates, label=alg_name)
    plt.legend()
    plt.xlabel("Iterate")
    plt.ylabel("|T_{m, m-1}|")
    plt.yscale("log")
    plt.title("|T_{m, m-1}| compared to iteration")
    plt.show()


if __name__ == '__main__':
    # part d
    A = construct_A(5)
    plot_superdiag_norms(A, shift=None, alg_name="Unshifted qr_alg_tri")

    # part e
    # --- 5x5 A in question
    alg_iterates, pure_iterates = concatenate(A)
    plot_iterates(alg_iterates, "Concatenations for A", pure_iterates)

    m = 12
    # --- random symmetric m x m matrix
    A = np.random.rand(m, m)
    A = A + A.T
    alg_iterates, pure_iterates = concatenate(A)
    plot_iterates(alg_iterates, "Concatenations for random matrix", pure_iterates)

    # part f
    wilk_iterates = concatenate(A, shift=True, compare_to_pure=False)
    plot_iterates(wilk_iterates, "Concatenations for random matrix", alg_name="Shifted qr_alg_tri")

    A = construct_A(5)
    wilk_iterates = concatenate(A, shift=True, compare_to_pure=False)
    plot_iterates(wilk_iterates, "Concatenations for A", alg_name="Shifted qr_alg_tri")

    # part g
    A = construct_A2(15)

    alg_iterates = concatenate(A, compare_to_pure=False)
    plot_iterates(alg_iterates, "Concatenations for A = D + O", alg_name="Unshifted qr_alg_tri")

    wilk_iterates = concatenate(A, shift=True, compare_to_pure=False)
    plot_iterates(wilk_iterates, "Concatenations for A = D + O", alg_name="Shifted qr_alg_tri")
