import matplotlib.pyplot as plt
from cla_utils import householder_qr, GS_classical, GS_modified


# compute the reduced Q for each algorithm
def reduced_Qs(A, p):
    fullhQ, hR = householder_qr(A)
    hQ = fullhQ[:, :p]
    cQ, cR = GS_classical(A)
    mQ, mR = GS_modified(A)
    return hQ, cQ, mQ


# plot the graph for part a
def plot_part_a(x_vals, Q):
    for i in range(6):
        lbl = "Column " + str(i+1)
        plt.plot(x_vals, Q[:, i], label=lbl)
    plt.title("Householder with p = 5")
    plt.grid()
    plt.axvline()
    plt.axhline()
    plt.legend()
    plt.show()


# function used to plot graphs in part b
def plot_plots(x_vals, p, Q, method):
    plt.show()
    for i in range(5):
        lbl = "Column " + str(p-5+i+1)
        plt.plot(x_vals, Q[:, -5+i], label=lbl)
    title = method + " with p = " + str(p)
    plt.title(title)
    plt.axvline()
    plt.axhline()
    plt.grid()
    plt.legend()
    plt.show()


# function to plot all graphs together
def plot_all(x_vals, p, hQ, cQ, mQ):
    plot_plots(x_vals, p, hQ, "Householder")
    plot_plots(x_vals, p, mQ, "MGS")
    plot_plots(x_vals, p, cQ, "CGS")