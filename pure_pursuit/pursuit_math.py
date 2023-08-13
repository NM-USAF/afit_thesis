import numpy as np
from . import utilities


def r_dot(phi, theta, mu):
    """
    dr/dt * 1/vp
    phi: the pursuer's heading
    theta: the evader's heading
    mu: speed ratio (v_E / v_P)
    """
    return mu * np.cos(phi - theta) - 1


def phi_dot(r, phi, theta, mu):
    """
    dphi/dt * 1/vp
    r: distance from pursuer to evader
    phi: the pursuer's heading
    theta: the evader's heading
    """
    return -mu * np.sin(phi - theta) / r


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
    mu: speed ratio (1 or 2)
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


def theta_max(mu):
    """
    the largest value of theta for which the evader can be caught at all.
    If `theta > theta_max`, then the evader will flee from the evader faster
    than the evader can travel - the distance from the pursuer to evader will
    only increase as time increases.

    mu: speed ratio
    """
    return np.arcsin(1 / mu)


def theta_min(lod):
    """
    The smallest value of theta for which the evader can escape at all. 
    If `theta < theta_min`, then the evader will run into the pursuer's capture
    range without the pursuer having moved at all.

    lod: capture radius `l` divided by initial distance `d`
    """
    return np.arctan(lod)


def r_min(theta, mu):
    """
    the closest the pursuer will get to the evader for mu >= 1
    theta: the evader's heading (constant)
    mu: speed ratio

    returns min(r(t)) / d
    """
    theta = utilities.fix_theta(theta)

    ct = np.cos(theta)
    st = np.sin(theta)
    
    r_m = (
        np.power(np.sqrt(1-1/mu**2)/ct, (1-mu)/mu)
        * np.power((1+st)/(1+1/mu), 1/mu)
    )

    # another way of constraining phi_m < pi/2
    invalid = theta > theta_max(mu)

    if isinstance(r_m, np.ndarray):
        r_m[invalid] = 1
    elif invalid:
        r_m = 1

    return r_m


def deriv_r_min_theta(theta, mu):
    """
    derivative of r_min with respect to theta.
    used in optimal_evader_heading.
    theta: evader's heading
    mu: speed ratio

    returns d/dtheta r_m / d
    """
    theta = utilities.fix_theta(theta)
    st = np.sin(theta)
    ct = np.cos(theta)
    m2 = mu**2

    a = mu * (st + 1) / (mu + 1) * np.sqrt(1 - 1/m2) * 1/ct
    b = 1 - mu * st
    c = np.sqrt(m2 - 1)

    deriv = np.power(a, 1/mu) * b / c

    # clear out invalid results in accordance with the implementation of r_min
    invalid = theta > theta_max(mu)
    deriv = np.where(invalid, 0, deriv)

    return deriv


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

