import numpy as np
from compmath.differential_equations import DifferentialEquation
from typing import Tuple
from typing import Callable


def euler(equation: DifferentialEquation, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a differential equation using the Euler method with fixed step size.

    Args:
        equation (DifferentialEquation): An instance of DifferentialEquation containing the initial conditions and function.
        h (float): The step size.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays, one for the x values and one for the corresponding y values.
    """
    x0, y0, a, b, f = equation.x0, equation.y0, equation.a, equation.b, equation.f
    xs = [x0]
    ys = [y0]
    x = x0
    y = y0

    while x < b:
        y += h * f(x, y)
        x += h
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


def extended_euler(equation: DifferentialEquation, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a differential equation using the Extended Euler method with fixed step size.

    Args:
        equation (DifferentialEquation): An instance of DifferentialEquation containing the initial conditions and function.
        h (float): The step size.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays, one for the x values and one for the corresponding y values.
    """
    x0, y0, a, b, f = equation.x0, equation.y0, equation.a, equation.b, equation.f
    xs = [x0]
    ys = [y0]
    x = x0
    y = y0

    while x < b:
        k1 = h * f(x, y)
        x_next = x + h
        y_next = y + k1
        k2 = h * f(x_next, y_next)
        y_next = y + 0.5 * (k1 + k2)

        xs.append(x_next)
        ys.append(y_next)
        x = x_next
        y = y_next

    return np.array(xs), np.array(ys)


def runge_kutta_4(equation: DifferentialEquation, h: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves a differential equation using the classical fourth-order Runge-Kutta (RK4) method with fixed step size.

    Args:
        equation (DifferentialEquation): An instance of DifferentialEquation containing the initial conditions and function.
        h (float): The step size.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays, one for the x values and one for the corresponding y values.
    """

    x0, y0, a, b, f = equation.x0, equation.y0, equation.a, equation.b, equation.f
    xs = [x0]
    ys = [y0]
    x = x0
    y = y0

    while x < b:
        k1 = h * f(x, y)
        k2 = h * f(x + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(x + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(x + h, y + k3)

        y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        xs.append(x + h)
        ys.append(y_next)
        x += h
        y = y_next

    return np.array(xs), np.array(ys)


def solve_adaptive_step_size(equation: DifferentialEquation, h: float, epsilon: float, method: Callable) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = method(equation, h)
    while True:
        xs_new, ys_new = method(equation, h / 2)
        if np.abs(ys[-1] - ys_new[-1]) < epsilon:
            break
        xs, ys = xs_new, ys_new

    return xs, ys
