from dataclasses import dataclass
from typing import Callable


@dataclass
class DifferentialEquation:
    """
    A class to represent a first-order differential equation.

    Attributes:
        x0 (float): The initial x value.
        y0 (float): The initial y value corresponding to x0.
        a (float): The start of the interval for solving the equation.
        b (float): The end of the interval for solving the equation.
        f (Callable[[float, float], float]): The function representing the differential equation dy/dx = f(x, y).
    """
    x0: float
    y0: float
    a: float
    b: float
    f: Callable[[float, float], float]
