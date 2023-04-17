import numpy as np


# the distance ratio from the pursuer to the evader
# phi: the pursuer's heading (changes over time)
# theta: the evader's heading (constant)
# mu: speed ratio (vE / vP)
# returns distance ratio s.t. r(phi, theta, mu) * d(0) = d(t)
def r(phi, theta, mu):
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi-theta)
    cp = np.cos(phi-theta)
    return np.power(sp/ct, (1-mu)/mu) * np.power((1+st)/(1+cp), 1/mu)


# heading of the pursuer (phi) at time of capture
# theta: evader's heading (constant)
# mu: speed ratio (1 or 2)
# dol: d/l; d is start distance, l is capture range
def phi_cap_1(theta, dol, mu=1):
    st = np.sin(theta)
    ct = np.cos(theta)
    if mu == 1:
        return theta + np.arccos((1+st)*dol - 1)
    elif mu == 2:
        # just let numpy solve the 4th order polynomial
        a = (dol**2) * (1 + st) * ct
        roots = np.array([
            np.roots([1, 2, 0, -2, (_a**2)-1])
            for _a in a
        ])
        where_valid = (np.isreal(roots)) & (roots > -1) & (roots < 1)

        # select the largest valid root as the solution
        # here x = cos(phi_cap - theta)
        # setting invalid roots to above the search range for argmin
        # is a bit of a hack, but I couldn't find a better way to do it.
        roots[~where_valid] = 1.1
        selector = np.argmin(roots, axis=1)
        x = np.take_along_axis(roots, selector[:,None], axis=1).flatten()

        assert len(x) == len(theta)

        # get rid of solutions that aren't valid
        where_valid = np.isreal(x) & (x > -1) & (x < 1)
        x[~where_valid] = None

        # recover desired value
        phi_cap = np.arccos(x) + theta
        return phi_cap
        

# time to capture
# theta: evader's heading (constant)
# mu: speed ratio (>= 1)
# lod: l/d; l is capture range, d is start distance
# returns t_c / d
def t_cap_1(theta, lod, mu):
    st = np.sin(theta)
    ct = np.cos(theta)
    if mu == 1:
        return 1/2 * (1 - lod - (1+st)/2 * np.log((2*lod - (1+st))/(1-st)))
    else:
        phi_c = phi_cap_1(theta, 1/lod, 2)
        psi = phi_c - theta
        sp = np.sin(psi)
        cp = np.cos(psi)
        tp2 = np.tan(psi/2)
        return 1/(1-mu**2) * ( (1+mu*st) - np.power(1+st, 1/mu)/np.power(ct, (1-mu)/mu) * (1 + mu*cp)/sp * np.power(tp2, 1/mu) )


# pursuer heading that minimizes distance for mu >= 1
# theta: evader's heading (constant)
# mu: speed ratio
# returns: phi_m 
def phi_min(theta, mu):
    return np.clip(np.arccos(1/mu) + theta, -np.pi/2, np.pi/2)


# the closest the persuer will get to the evader for mu >= 1
# theta: the evader's heading (constant)
# mu: speed ratio
# returns: r_m / d
def r_min(theta, mu):
    ct = np.cos(theta)
    st = np.sin(theta)
    return (
        np.power(np.sqrt(1-1/mu**2)/ct, (1-mu)/mu)
        * np.power((1+st)/(1+1/mu), 1/mu)
    )

