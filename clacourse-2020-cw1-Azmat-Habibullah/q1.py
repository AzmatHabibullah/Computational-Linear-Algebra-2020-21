from numpy import *
from cla_utils import householder_qr
np.set_printoptions(linewidth=320)

A = load('values.npy')


if __name__ == "__main__":
    # part a
    Q, R = householder_qr(A)
    print(np.around(Q, 4))
    print(np.around(R, 4))

    # part b
    R_diag = np.diag(R)

    # check the first four columns of Q are important
    print(np.abs(R_diag)[:5] > 10**-6)

    # check only the first four columns of Q are important
    print(sum(np.abs(R_diag) > 10**-6))

    # check the errors for varying number of lines used
    errors = [np.linalg.norm(Q[:, :i].dot(R[:i, :]) - A) ** 2 for i in range(10)]
    print(errors)
