import numpy as np
from .pursuit_math import *
from scipy.optimize import minimize_scalar, differential_evolution


def can_escape_simple(
    mu_left, mu_right,
    angle_between
):
    """
    Can the evader escapt by simply accepting a heading such that:
    `theta_l > theta_max_l & theta_r > theta_max_r`
    or: such that neither of the pursuers can actually get any closer to the
    evader as time increases.

    returns (can_escape :: bool, theta_l :: float)
    """
    theta_max_l = theta_max(mu_left)
    theta_max_r = theta_max(mu_right)

    angle_between_max = np.pi - theta_max_l - theta_max_r

    # evader is pointing away from the pursuers
    angle_between_min = angle_between

    can_escape = angle_between_max > angle_between_min

    # half of the angle between pursuers is as good as any other heading
    # within the range where theta_l > theta_max_l, theta_r > theta_max_r
    theta_l = (np.pi - angle_between) / 2

    return can_escape, theta_l


def optimize_evader_heading_newton(
    theta_left,
    evader_distance_ratio, 
    lod_left, lod_right, 
    mu_left, mu_right, 
    angle_between,         
):
    """
    Runs one step of newton's method to optimize the evader's heading relative
    to the pursuer on the left in an engagement against two pursuers.

    theta_left: evader's heading against the left pursuer
    evader_distance_ratio: distance to left evader divided by distance to
                           the right evader
    lod_left: capture radius over distance of the left pursuer
    lod_right: capture radius over distance of the right pursuer
    angle_between: angle between the two evaders
    mu: speed ratio (identical for both evaders)
    """
    
    # goal: find theta_l s.t. (r_min_l*kd - lod_left*kd) == (r_min_r - lod_right) > 0
    # or s.t. (r_min_l*kd - l_left*kd) - (r_min_r - l_right) = 0
    # where r_min_x = r(phi_m_x)

    # gamma used in place of angle_between, and kd in place of 
    # evader_distance_ratio for comments below
    kd = evader_distance_ratio

    # so f = r_min_l*kd - r_min_r + l_right - l_left*kd
    # in th_l_n+1 = th_l_n - f(th_l_n) / f'(th_l_n)
    # so f' = d(r_min_l)/d(th_l)*kd - d(r_min_r)/d(th_r) * d(th_r)/d(th_l)
    # where d(th_r) / d(th_l) = -1 from note in function description
    # so f' = d(r_min_l)/d(th_l) + d(r_min_r)/d(th_r)

    th_r = angle_between - theta_left - np.pi

    r_l = r_min(theta_left, mu_left)
    r_r = r_min(th_r, mu_right)

    f_th_l = r_l*kd - r_r + lod_right - lod_left*kd
    df_th_l = (
        deriv_r_min_theta(theta_left, mu_left)*kd 
        + deriv_r_min_theta(th_r, mu_right)
    )
    
    # if the derivative is zero, we are either at a minimum already
    # or we are at the plateau where simple escape is possible
    if np.all(df_th_l == 0):
        return theta_left

    # intelligently clip to valid bounds
    return utilities.fix_theta(theta_left - f_th_l / df_th_l)


def optimal_evader_heading_newton(
    evader_distance_ratio, 
    lod_left, lod_right, 
    mu_left, mu_right, 
    angle_between, 
    n_iters=10
):
    """
    Newton's method approximation of the optimal constant evader heading given
    the engagement conditions.

    evader_distance_ratio: distance to left evader divided by distance to
                           the right evader
    lod_left: capture radius over distance of the left pursuer
    lod_right: capture radius over distance of the right pursuer
    angle_between: angle between the two evaders
    mu: speed ratio (identical for both evaders)
    n_iters: number of iterations of newton's method to use. default is 5.
             Emperically, typically no more than 5 iterations are needed for
             acceptable precision.

    NOTE: this function will diverge and return NAN on occasion if
          angle_between is near pi or near zero, mu is large, or 
          one pursuer is much closer than the other.

    returns theta_l
        note: theta_r = angle_between - theta_l - pi
    """

    # psi(0) = pi/2 - theta (phi(0) = pi/2)
    # psi_l(0) + psi_r(0) = 2pi - angle_between
    # pi/2 - theta_l + pi/2 - theta_r = 2pi - angle_between

    # initial guess: half of the angle between pursuers
    # alpha_0 = gamma / 2 & theta_l = alpha - pi/2 
    # -> theta_l_0 = (gamma - pi) / 2
    th_l = (angle_between - np.pi) / 2

    for _ in range(n_iters):
        th_l = optimize_evader_heading_newton(
            th_l, 
            evader_distance_ratio,
            lod_left, lod_right,
            mu_left, mu_right,
            angle_between
        )

    return th_l


def capture_heading_binary(
    mu, lod, initial_max=np.pi/2, initial_min=-np.pi/2, n_iters=10
):
    """
    Use a binary search to find the heading at which the evader is just barely
    captured. Any heading larger than this, and the evader will escape - any 
    lower, and the evader will definitely be captured.

    mu: evader/pursuer speed ratio
    lod: pursuer capture radius over initial distance
    """
    th_max = initial_max
    th_min = initial_min
    for _ in range(n_iters):
        th_mid = (th_max + th_min) / 2
        capture_margin = r_min(th_mid, mu) - lod

        th_max = np.where(capture_margin > 0, th_mid, th_max)
        th_min = np.where(capture_margin < 0, th_mid, th_min)

    return th_mid


def negative_relu(x):
    return np.where(x < 0, x, 0)


def optimal_evader_heading_scipy(
    lods,
    mus,
    headings,
    method="multiple_local",
    n_guesses=None
):
    """
    Finds the optimal constant evader heading theta by minimizing the negative
    of the closest that either evader will come to the pursuer.

    returns: theta, min(r_min - l/d for each pursuer)
    """

    def to_optimize(theta):
        """
        theta = theta_left
        """

        pursuer_thetas, _ = utilities.engagement_heading(theta, headings)
        rmins = r_min(pursuer_thetas, mus) - lods

        return -np.min(rmins)
    
    if method == "differential_evolution":
        result = differential_evolution(
            to_optimize, 
            [[-np.pi, np.pi]],
            popsize=len(lods+1),
            seed=0
        )
        
        return result.x[0], result.fun

    if method == "multiple_local":
        if not n_guesses:
            # number of guesses = number of pursuers + 1
            n_guesses = len(lods) + 1

        bounds = np.array([
            np.arange(0, n_guesses),
            np.arange(1, n_guesses+1)
        ]).astype(float).T / (n_guesses) * 2 * np.pi

        # run many local optimizations
        results = [minimize_scalar(
            to_optimize,
            method="bounded",
            bounds=b
        ) for b in bounds]

        res_min = min(results, key=lambda r: r.fun)

        return res_min.x, res_min.fun


def optimal_evader_heading(
    evader_distance_ratio, 
    lod_left, lod_right, 
    mu_left, mu_right, 
    angle_between, 
    method="newton",
    **kwargs
):
    # First test: simple escape, no optimization necessary
    can, theta_l_simple = can_escape_simple(mu_left, mu_right, angle_between)

    if np.all(can):
        return theta_l_simple

    if method == "newton":
        theta_l_optimize = optimal_evader_heading_newton(
            evader_distance_ratio,
            lod_left,
            lod_right,
            mu_left,
            mu_right,
            angle_between,
            **kwargs
        )

        if isinstance(can, np.ndarray):
            return np.where(can, theta_l_simple, theta_l_optimize)
        elif can:
            return theta_l_simple
        else:
            return theta_l_optimize
    elif method == "scipy":
        theta_l_optimize = optimal_evader_heading_scipy(
            evader_distance_ratio,
            lod_left,
            lod_right,
            mu_left,
            mu_right,
            angle_between
        )

        return theta_l_optimize
