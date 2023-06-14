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

