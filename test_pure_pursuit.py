import pytest
import numpy as np
import pure_pursuit as pp

eps_input = 0.001
eps_output = 1e-6

lod_min = 0.01
lod_max = 0.99
mus = np.linspace(1+eps_input, 10)
lod = np.linspace(lod_min, lod_max)

def angles_all_close(a1, a2, **kwargs):
    angle_differences = abs(a1 - a2) % (2*np.pi)

    # we may get numbers that are almost 2pi. The mod operator above
    # will not return a negative number, but the same positive number.
    # So we also use abs(angle_differences - 2pi) to catch those cases.
    neg_angle_differences = abs(angle_differences - 2*np.pi)
    angle_differences = np.where(
        neg_angle_differences < angle_differences,
        neg_angle_differences,
        angle_differences
    )
    return np.allclose(angle_differences, 0, **kwargs)


def random_world_pursuers(n):
    gammas = np.random.uniform(-np.pi, np.pi, n)
    mus = np.random.uniform(1, 10, n)
    distances = np.random.uniform(2, 10, n)
    capture_radius = np.random.uniform(1, distances-1, n)
    
    return gammas, mus, distances, capture_radius


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
                

# test that fix_theta always puts theta within [-pi/2, pi/2]
def test_fix_theta():
    thetas = np.linspace(-20, 20, 4000)
    fixed = pp.utilities.fix_theta(thetas)
    assert np.all((fixed <= np.pi/2) & (fixed >= -np.pi/2))


# test that the capture heading theta_c is where r_min/d == l/d
def test_capture_heading():
    for mu in mus:
        theta_cap = pp.capture_heading(mu, lod, n_iters=30)
        fail = np.isnan(theta_cap)
        assert not np.any(np.isnan(theta_cap)), f"mu={mu}, lod={lod[fail]}"

        r_theta_cap = pp.r_min(theta_cap, mu)

        assert np.allclose(r_theta_cap, lod, atol=eps_output), theta_cap


# test that converting from a world-frame evader heading to a pursuer-frame
# evader heading works
def test_heading_frame_conversions():
    for pursuer_angle in np.linspace(-np.pi, np.pi):
        evader_angle = np.linspace(-np.pi + eps_input, np.pi - eps_input)
        theta_p, direction = pp.utilities.engagement_heading(evader_angle, pursuer_angle)
        where_should_be_negative = abs(pursuer_angle - evader_angle) < np.pi/2
        assert np.all(theta_p[where_should_be_negative] < 0)

        theta_w = pp.utilities.world_heading(theta_p, pursuer_angle, direction=direction)
        assert angles_all_close(evader_angle, theta_w, atol=eps_output), f"gamma={pursuer_angle}, theta_w={evader_angle}, theta_p={theta_p}, dir={direction}"


# test that the evader can escape for angles given by the capture angle
# range subtraction algorithm
def test_capture_angle_binary_interval():
    n_tests = 10
    for i in range(n_tests):
        n_pursuers = int(np.random.uniform(2, 10))
        g, m, d, l = random_world_pursuers(n_pursuers)
        interval, escape = pp.optimal_evader_heading_capture_angle(l/d, m, g)
        
        theta_starts, _ = pp.utilities.engagement_heading(interval[0] + eps_input, g)
        theta_ends, _ = pp.utilities.engagement_heading(interval[1] - eps_input, g)
        theta_mids, _ = pp.utilities.engagement_heading((interval[0]+interval[1])/2, g)

        for theta in [theta_starts, theta_ends, theta_mids]:
            cap_margins = pp.r_min(theta, m) - l/d
            if escape:
                assert np.all(cap_margins > 0)
            else:
                assert np.any(cap_margins < 0)