import numpy as np

def law_of_cosines_angle(a, b, c):
    """
    given the three sides of a triangle, finds the angle opposite of side `c`
    a, b: off-sides of the triangle
    c: side of the triangle opposite the angle of interest
    returns: the angle opposite of side c
    """

    # https://en.wikipedia.org/wiki/Law_of_cosines#Use_in_solving_triangles
    return np.arccos((a**2 + b**2 - c**2) / (2*a*b))


def wrap(value, max_val):
    """
    wraps value to the range [-max_val, max_val]
    """
    return (value + max_val) % (2 * max_val) - max_val


def mirror(value, point):
    """
    mirrors a value across a point. mirroring across 0 is the same as negation.
    example: mirror(1, 3) = 5
    """
    return point - (value - point)


def fix_theta(theta):
    """
    for any theta in [-pi, pi] in the engagement frame, maps that theta to an
    equivalent theta  in the range [-pi/2, pi/2] that can be used in the pure 
    pursuit equations
    """
    wrapped = wrap(theta, np.pi)
    mirr_hi = np.where(wrapped > np.pi/2, mirror(wrapped, np.pi/2), wrapped)
    mirr_lo = np.where(mirr_hi < -np.pi/2, mirror(mirr_hi, -np.pi/2), mirr_hi)
    return mirr_lo


def compose_linear(m1, b1, m2, b2):
    return m1*m2, m1*b1 + b2


def negative_relu(x):
    return np.where(x < 0, x, 0)


def engagement_heading(evader_heading, pursuer_angle):
    """
    Given the evader's heading in the world frame and the heading to a pursuer
    in that frame, calculates a theta value that can be used in the pure 
    pursuit equations.

    returns (theta, direction)
    direction=-1 means the evader's trajectory is to the right of the pursuer
    instead of to the left. Necessary to convert back to world coordinates.
    """
    psi = wrap(pursuer_angle - evader_heading, np.pi)
    theta = fix_theta(abs(psi) - np.pi/2)
    direction = np.where(psi < 0, 1, -1)
    return theta, direction

def world_heading(evader_heading, pursuer_angle, direction=1):
    """
    Given the evader's heading in the pursuer frame and the heading to that
    pursuer with which side of the pursuer it's on (left=1, right=-1),
    calculates the world frame heading

    returns (theta_w)
    """
    return wrap(pursuer_angle + direction * (evader_heading + np.pi/2), np.pi)
