import pytest
import numpy as np
import pure_pursuit as pp

eps_input = 0.001
eps_output = 1e-6

lod_min = 0.01
lod_max = 0.99

# test that the analytical solution to r_min and r(phi_min) are the same
def test_phi_r_min():
    mus = np.linspace(1, 10)
    thetas = np.linspace(-np.pi/2+eps_input, np.pi/2-eps_input, 100)

    for m in mus:
        phi_m = pp.phi_min(thetas, m)
        r_m = pp.r(phi_m, thetas, m)
        assert np.allclose(pp.r_min(thetas, m), r_m, atol=eps_output)

# test that r(phi_cap) == l/d 
def test_phi_cap():
    mus = [1, 2]
    lod = np.linspace(lod_min, lod_max)

    theta_max = np.arcsin(2*lod - 1)
    thetas = np.linspace(-np.pi/2+0.1, theta_max)

    for mu in mus:
        for i, l_over_d in enumerate(lod):
            phi_cap = pp.phi_cap_1(thetas[:,i], l_over_d, mu)

            # drop nan results from phi_cap
            thetas_trimmed = thetas[:,i][~np.isnan(phi_cap)]
            phi_cap = phi_cap[~np.isnan(phi_cap)]

            r_phi_cap = pp.r(phi_cap, thetas_trimmed, mu)

            assert np.allclose(l_over_d, r_phi_cap, atol=eps_output)
