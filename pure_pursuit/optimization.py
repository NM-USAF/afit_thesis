import numpy as np
from .pursuit_math import *


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


def optimize_evader_heading(
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


def optimal_evader_heading_newtons(
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

    # initial guess: half of the angle between pursuers
    # alpha_0 = gamma / 2 & theta_l = alpha - pi/2 
    # -> theta_l_0 = (gamma - pi) / 2
    th_l = (angle_between - np.pi) / 2

    for _ in range(n_iters):
        th_l = optimize_evader_heading(
            th_l, 
            evader_distance_ratio,
            lod_left, lod_right,
            mu_left, mu_right,
            angle_between
        )

    return th_l


def optimal_evader_heading(
    evader_distance_ratio, 
    lod_left, lod_right, 
    mu_left, mu_right, 
    angle_between, 
    **kwargs
):
    # First test: simple escape, no optimization necessary
    can, theta_l_simple = can_escape_simple(mu_left, mu_right, angle_between)

    if np.all(can):
        return theta_l_simple

    theta_l_optimize = optimal_evader_heading_newtons(
        evader_distance_ratio,
        lod_left,
        lod_right,
        mu_left,
        mu_right,
        angle_between,
        **kwargs
    )

    if isinstance(theta_l_simple, np.ndarray):
        # something
        return np.where(can, theta_l_simple, theta_l_optimize)
    elif can:
        return theta_l_simple
    else:
        return theta_l_optimize
    
    