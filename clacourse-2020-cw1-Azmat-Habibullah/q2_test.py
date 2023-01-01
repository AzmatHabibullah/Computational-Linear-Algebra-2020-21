import pytest
from q2_new import *
from numpy import *


@pytest.mark.parametrize('deg_poly', range(11))
def test_interpolation(deg_poly):
    x = arange(-1, 1.1, 0.2)
    f = 0.0 * x
    f[3:6] = 1

    __, my_coefficients = compute_coefficients(x, f, deg_poly)
    flipped_numpy_coefficients = polyfit(x, f, deg_poly)
    numpy_coefficients = np.flip(flipped_numpy_coefficients)

    # verify correct dimensions
    assert(my_coefficients.size == deg_poly + 1)

    # verify our coefficients are close to those of numpy's implementation
    # this is a valid test as interpolating polynomials of the same degree for a given dataset are unique
    assert(np.max(np.abs(numpy_coefficients - my_coefficients)) < 10**-10)

    # when the same degree
    if deg_poly == 10:
        vandermonde = np.vander(x, deg_poly + 1, increasing=True)
        assert(np.max(np.abs(f - vandermonde.dot(my_coefficients))) < 10**-10)
