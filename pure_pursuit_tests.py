import pytest
import numpy as np
import pure_pursuit as pp

eps_input = 0.001
eps_output = 1e-6

# test that r_min and r(phi_min) are the same
def test_phi_r_min():
    mus = np.linspace(1, 10)
    thetas = np.linspace(-np.pi/2+eps_input, np.pi/2-eps_input, 100)

    for m in mus:
        phi_m = pp.phi_min(thetas, m)
        r_m = pp.r(phi_m, thetas, m)
        assert np.allclose(pp.r_min(thetas, m), r_m, atol=eps_output)

# test that r(phi_cap) == l/d