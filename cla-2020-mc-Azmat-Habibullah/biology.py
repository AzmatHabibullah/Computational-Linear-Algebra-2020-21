"""
Functions that were investigated in the cell division example
"""


def mathbio_nullcline(a):
    def function(x):
        return x/(a + x**2)
    return function


def mathbio_integral(a, c):
    def function(x):
        return (a - x**2)/((c - x)*(a + x**2)**2)
    return function