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

def inorder(l):
    return l == sorted(l)

def subtract_range(start1, end1, start2, end2):
    """
    Given an range [start1, end1], subtracts the set of numbers that fall
    within (start2, end2) to return a new set of numbers. The resulting set 
    may be discontinuous.
    """
    # start1 = a, end1 = b, start2 = c, end2 = d
    
    assert start1 < end1 and start2 < end2
    
    # case 1: a < b < c < d or c < d < a < b -> { [a, b] }
    if (
        inorder([start1, end1, start2, end2]) 
        or inorder([start2, end2, start1, end1])
    ):
        return { (start1, end1) }
    # case 2: c < a < b < d -> {}
    elif inorder([start2, start1, end1, end2]):
        return set()
    # case 3: c < a < d < b -> { [d, b] }
    elif inorder([start2, start1, end2, end1]):
        return { (end2, end1) }
    # case 4: a < c < b < d -> { [a, c] }
    elif inorder([start1, start2, end1, end2]):
        return { (start1, start2) }
    # case 5: a < c < d < b -> { [a, c], [d, b] }
    elif inorder([start1, start2, end2, end1]):
        return { (start1, start2), (end2, end1) }


def subtract_ranges(start_range, ranges):
    """
    Subtracts multiple sets of numbers `ranges` from the initlal range
    `start_range`. 
    """
    result = { start_range }
    for r in ranges:
        next_result = set()
        for o in result:
            rs, re = r
            os, oe = o
            next_result = next_result | subtract_range(os, oe, rs, re)
        result = next_result
    return result


def wrap_range(start_range, end_range, wrap):
    """
    Wraps a range of numbers around a boundary point `wrap` where `-wrap=wrap`.
    The resulting set of numbers may be discontinuous on a number line.
    """
    assert start_range < end_range
    
    wrap2 = 2*wrap
    
    if end_range < -wrap:
        return { (start_range+wrap2, end_range+wrap2) }
    elif start_range > wrap:
        return { (start_range-wrap2, end_range-wrap2) }
    elif start_range < -wrap and end_range > -wrap:
        return { (-wrap, end_range), (start_range+wrap2, wrap) }
    elif start_range < wrap and end_range > wrap:
        return { (start_range, wrap), (-wrap, end_range-wrap2) }
    return { (start_range, end_range) }