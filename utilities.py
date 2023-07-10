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
    wrapped = wrap(theta, np.pi)
    mirr_hi = np.where(wrapped > np.pi/2, mirror(wrapped, np.pi/2), wrapped)
    mirr_lo = np.where(mirr_hi < -np.pi/2, mirror(mirr_hi, -np.pi/2), mirr_hi)
    return mirr_lo


def compose_linear(m1, b1, m2, b2):
    return m1*m2, m1*b1 + b2