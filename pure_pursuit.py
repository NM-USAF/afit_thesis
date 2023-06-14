import numpy as np


def r(phi, theta, mu):
    """
    the distance ratio from the pursuer to the evader
    phi: the pursuer's heading (changes over time)
    theta: the evader's heading (constant)
    mu: speed ratio (vE / vP)

    returns distance ratio s.t. r(phi, theta, mu) * d(0) = d(t)
    """
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi-theta)
    cp = np.cos(phi-theta)
    return np.power(sp/ct, (1-mu)/mu) * np.power((1+st)/(1+cp), 1/mu)


def phi_cap_1(theta, lod, mu=1):
    """
    heading of the pursuer (phi) at time of capture
    theta: evader's heading (constant)
    mu: speed ratio (1 or 2)
    lod: l/d; d is start distance, l is capture range
    """
    st = np.sin(theta)
    ct = np.cos(theta)
    if mu == 1:
        return theta + np.arccos((1+st)/lod - 1)
    elif mu == 2:
        # just let numpy solve the 4th order polynomial
        a = (1 + st) / (lod**2)  * ct
        roots = np.array([
            np.roots([1, 2, 0, -2, (_a**2)-1])
            for _a in a
        ])

        # psi can't be greater than pi/2-theta or less than 0
        bound = np.cos(np.pi/2-theta)
        where_valid = (
            (np.isreal(roots)) & 
            (roots > bound[:,None]) & 
            (roots < 1)
        )

        # select the smallest valid root as the solution.
        # the smallest root will match the largest angle difference,
        # which would be the first occurence of the evader-pursuer distance
        # crossing the capture radius. Because the evader starts out
        # of range of the pursuer, this is when the evader first crosses
        # into the capture range of the pursuer.
        # here x = cos(phi_cap - theta)
        
        # setting invalid roots to above the search range for argmin
        # is a bit of a hack, but I couldn't find a better way to do it.
        roots[~where_valid] = 1.1
        selector = np.argmin(roots, axis=1)
        x = np.take_along_axis(roots, selector[:,None], axis=1).flatten()

        assert len(x) == len(theta)

        # get rid of solutions that aren't valid
        where_valid = np.isreal(x) & (x > bound) & (x < 1)
        x[~where_valid] = None

        # recover desired value
        phi_cap = np.arccos(x) + theta
        return phi_cap
        

def t_cap_1(theta, lod, mu):
    """
    time to capture
    theta: evader's heading (constant)
    mu: speed ratio (>= 1)
    lod: l/d; l is capture range, d is start distance

    returns t_c / d
    """
    st = np.sin(theta)
    ct = np.cos(theta)
    if mu == 1:
        return 1/2 * (1 - lod - (1+st)/2 * np.log((2*lod - (1+st))/(1-st)))
    else:
        phi_c = phi_cap_1(theta, lod, 2)
        psi = phi_c - theta # psi > 0
        sp = np.sin(psi)
        cp = np.cos(psi)
        tp2 = np.tan(psi/2)
        return 1/(1-mu**2) * ( (1+mu*st) - np.power(1+st, 1/mu)/np.power(ct, (1-mu)/mu) * (1 + mu*cp)/sp * np.power(tp2, 1/mu) )


def phi_min(theta, mu):
    """
    the pursuer heading at which distance to the evader is minimized
    theta: evader's heading (constant)
    mu: speed ratio

    returns phi_m 
    """
    if mu < 1:
        return theta
    else:
        return np.clip(np.arccos(1/mu) + theta, -np.pi/2, np.pi/2)


def r_min(theta, mu):
    """
    the closest the pursuer will get to the evader for mu >= 1
    theta: the evader's heading (constant)
    mu: speed ratio

    returns r_m / d
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    
    r_m = (
        np.power(np.sqrt(1-1/mu**2)/ct, (1-mu)/mu)
        * np.power((1+st)/(1+1/mu), 1/mu)
    )

    # another way of constraining phi_m < pi/2
    invalid = theta > np.arcsin(1/mu)

    if isinstance(r_m, np.ndarray):
        r_m[invalid] = 1
    elif invalid:
        r_m = 1

    return r_m


def deriv_r_min_theta(theta, mu):
    """
    derivative of r_min with respect to theta. Used for finding optimal evader
    headings with Newton's method.
    theta: evader's heading
    mu: speed ratio

    returns d/dtheta r_m / d
    """
    st = np.sin(theta)
    ct = np.cos(theta)
    m2 = mu**2

    a = mu * (st + 1) / (mu + 1) * np.sqrt(1 - 1/m2) * 1/ct
    b = 1 - mu * st
    c = np.sqrt(m2 - 1)

    deriv = np.power(a, 1/mu) * b / c

    # clear out invalid results
    invalid = theta > np.arcsin(1 / mu)
    if isinstance(deriv, np.ndarray):
        deriv[invalid] = 0
    else:
        deriv = 0

    return deriv


def optimal_evader_heading(
    d_left, d_right, angle_between, mu, n_iters=3
):
    """
    d_left: distance to the left evader
    d_right: distance to the right evader
    angle_between: angle between the two evaders
    mu: speed ratio (identical for both evaders)
    n_iters: number of iterations of newton's method to use. default is 3.
             Emperically, typically no more than 3 iterations are needed for
             acceptable precision.

    NOTE: this function will diverge and return NAN on occasion if
          angle_between is near pi or near zero, mu is large, or 
          one pursuer is much closer than the other.
          this is (mostly) guaranteed to work for all values within:
          - 0.01 < d_left / d_right < 0.01
          - 1.001 < mu < 10
          - 0.001 < angle_between < pi-0.8


    returns theta_l
        note: theta_r = angle_between - theta_l - pi
    """

    # gamma used in place of angle_between for comments below

    # initial guess: half of the angle between pursuers
    # alpha_0 = gamma / 2 & theta_l = alpha - pi/2 
    # -> theta_l_0 = (gamma - pi) / 2
    th_l = (angle_between - np.pi) / 2

    for _ in range(n_iters):
        # goal: find theta_l such that r_min_l == r_min_r
        # or s.t. r_min_l - r_min_r = 0
        # where r_min_x = r(phi_m_x)/d_x * d/x

        # so f = r_min_l - r_min_r 
        # in th_l_n+1 = th_l_n - f(th_l_n) / f'(th_l_n)
        # so f' = d(r_min_l)/d(th_l) - d(r_min_r)/d(th_r) * d(th_r)/d(th_l)
        # where d(th_r) / d(th_l) = -1 from note in function description
        # so f' = d(r_min_l)/d(th_l) + d(r_min_r)/d(th_r)

        th_r = angle_between - th_l - np.pi

        f_th_l = r_min(th_l, mu)*d_left - r_min(th_r, mu)*d_right
        df_th_l = (
            deriv_r_min_theta(th_l, mu)*d_left 
            + deriv_r_min_theta(th_r, mu)*d_right
        )

        th_l -= f_th_l / df_th_l

        # intelligently clip to valid bounds
        th_l = np.fmod(th_l, 2*np.pi)
        th_l[th_l > np.pi] -= 2*np.pi
        th_l[th_l < -np.pi] += 2*np.pi

        th_l = np.where(th_l < -np.pi/2, -th_l - np.pi, th_l)

        max_th_l = angle_between - np.pi/2
        th_l = np.where(th_l > max_th_l, 3*max_th_l - 2*th_l, th_l)

    return th_l


def polygon_formation_capture_ratio_d(mu, n):
    """
    distance ratio at which capture occurs when `n` pursuers 
    are surrounding an evader in a regular polygon shape 
    mu: evader/pursuer speed ratio
    n: number of sides to the polygon
    returns: crit_value where l/d > crit_value -> capture will occur
    """
    angle = np.pi / n
    ca = np.cos(angle)
    sa = np.sin(angle)
    return (
        np.power(np.sqrt(1-1/mu**2)/sa, (1-mu)/mu)
        * np.power((1-ca)/(1+1/mu), 1/mu)
    )


def polygon_formation_capture_ratio_a(mu, n):
    """
    side length to capture radius ratio at which capture occurs when `n` 
    pursuers are surrounding an evader in a regular polygon shape, where 
    `a` is the side length of the polygon formation
    mu: evader/pursuer speed ratio
    n: number of sides to the polygon
    returns: crit_value where l/a > crit_value -> capture will occur
    """
    a_d_ratio = 2 * np.sin(np.pi / n)
    return polygon_formation_capture_ratio_d(mu, n) / a_d_ratio

