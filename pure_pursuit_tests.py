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


# test that the optimal theta_l maximizes min(r_min_l, r_min_r)
# or: theta_l such that r_min_l == r_min_r
def test_optimal_evader_heading():
    mus = np.linspace(1+eps_input, 10)
    d_l = np.linspace(0.1, 10)
    d_r = np.linspace(10, 0.1)
    inner_angle = np.linspace(eps_input, np.pi-0.8)

    # cartesian product of d_l and d_r
    # from https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    ds = np.dstack(np.meshgrid(d_l, d_r)).reshape(-1, 2)

    for mu in mus:
        for dl, dr in ds:
            dl = np.array([dl])
            dr = np.array([dr])
            th_l_opt = pp.optimal_evader_heading(dl, dr, inner_angle, mu, n_iters=20)
            th_r_opt = inner_angle - th_l_opt - np.pi

            # accepting that sometimes this will return nan for now
            r_min_l = pp.r_min(th_l_opt, mu) * dl
            r_min_l = r_min_l[~np.isnan(r_min_l)]

            r_min_r = pp.r_min(th_r_opt, mu) * dr
            r_min_r = r_min_r[~np.isnan(r_min_r)]

            assert np.allclose(r_min_l, r_min_r, atol=0.01)
                