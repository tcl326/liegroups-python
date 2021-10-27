"""
Collection of utility functions
"""
from typing import Tuple

import math
import random


def normalize_range(theta: float, a: float = -math.pi, b: float = math.pi) -> float:
    # Normalize the value to range between (a, b]
    # See: https://stackoverflow.com/questions/24234609/standard-way-to-normalize-an-angle-to-%CF%80-radians-in-java/24234924
    range = b - a
    return theta - (range) * math.floor((theta - a) / (range))


def uniform_sampling_n_ball_muller(n: int) -> Tuple[float]:
    """
    # Sample a point uniformly inside a unit N-ball using Muller's method
    # http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    """
    points = tuple(random.gauss(0, 1.0) for _ in range(n))
    r = random.random() ** (1 / n)
    n = norm(points)
    return tuple(r * p / n for p in points)


def norm(vector: Tuple[float]) -> float:
    """
    Compute the l2 norm of a given vector
    """
    return math.sqrt(sum([v * v for v in vector]))


def clip(value, min_v, max_v):
    """
    Clip the given value to be within a range
    https://stackoverflow.com/questions/9775731/clamping-floating-numbers-in-python
    """
    return max(min(value, max_v), min_v)
