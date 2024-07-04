from typing import Any

import numpy as np
import numpy.typing as npt


def add_docstring(docstring: str) -> Any:
    """Decorator to add a docstring to a class.

    Args:
        docstring (str): Docstring to apply to class.
    """

    def apply_docstring(cls: Any) -> Any:
        cls.__doc__ = docstring
        return cls

    return apply_docstring


def assert_gt(x: Any, lb: Any, name: str, strict: bool = False) -> None:
    """Assert that x is greater than (or equal to) a lower bound.

    Args:
        x (Any): Value to check.
        lb (Any): Lower bound.
        name (str): Name of value for throwing error.
        strict (bool, optional): Whether the inequality is strict. Defaults to False.

    Raises:
        ValueError: When x <= lb (strict=True), or x < lb (strict=False).
    """
    if strict:
        if x <= lb:
            raise ValueError(f"{name} must be > {lb}")
    else:
        if x < lb:
            raise ValueError(f"{name} must be >= {lb}")


def assert_lt(x: Any, ub: Any, name: str, strict: bool = False) -> None:
    """Assert that x is less than (or equal to) an upper bound.

    Args:
        x (Any): Value to check.
        ub (Any): Upper bound.
        name (str): Name of value for throwing error.
        strict (bool, optional): Whether the inequality is strict. Defaults to False.

    Raises:
        ValueError: When x >= ub (strict=True), or x > ub (strict=False).
    """
    if strict:
        if x >= ub:
            raise ValueError(f"{name} must be < {ub}")
    else:
        if x > ub:
            raise ValueError(f"{name} must be <= {ub}")


def assert_positive(x: Any, name: str) -> None:
    """Assert that x > 0

    Args:
        x (Any): Value to check
        name (str): Name of value for throwing error

    Raises:
        ValueError: If x <= 0
    """
    assert_gt(x, 0, name, strict=True)


def assert_nonnegative(x: Any, name: str) -> None:
    """Assert that x >= 0

    Args:
        x (Any): Value to check
        name (str): Name of value for throwing error

    Raises:
        ValueError: If x < 0
    """
    assert_gt(x, 0, name, strict=False)


def assert_type(x: Any, tp: Any, name: str) -> None:
    """Assert that x is of type tp

    Args:
        x (Any): Value to check
        tp (Any): Type to check
        name (str): Name of value for throwing error

    Raises:
        TypeError: If not isinstance(x, tp)
    """
    if not isinstance(x, tp):
        raise TypeError(f"{name} must be of type {tp}. Got type {type(x)}")


def assert_gt_numpy(
    x: npt.NDArray, lb: np.ScalarType, name: str, strict: bool = False  # type: ignore
) -> None:
    """Assert that all elements of x are greater than (or equal to) a lower bound.

    Args:
        x (npt.NDArray): Array to check.
        lb (np.ScalarType): Lower bound.
        name (str): Name of value for throwing error.
        strict (bool, optional): Whether the inequality is strict. Defaults to False.

    Raises:
        ValueError: When x <= lb (strict=True), or x < lb (strict=False).
    """
    if strict:
        if np.all(x <= lb):
            violated_indices = (x <= lb).nonzero()
            raise ValueError(
                f"All values of {name} must be > {lb}. Violated at indices {violated_indices}."
            )
    else:
        if np.all(x < lb):
            violated_indices = (x < lb).nonzero()
            raise ValueError(
                f"All values of {name} must be >= {lb}. Violated at indices {violated_indices}."
            )


def assert_lt_numpy(
    x: npt.NDArray, ub: np.ScalarType, name: str, strict: bool = False  # type: ignore
) -> None:
    """Assert that all elements of x are less than (or equal to) a lower bound.

    Args:
        x (npt.NDArray): Array to check.
        ub (np.ScalarType): Upper bound.
        name (str): Name of value for throwing error.
        strict (bool, optional): Whether the inequality is strict. Defaults to False.

    Raises:
        ValueError: When x >= ub (strict=True), or x > ub (strict=False).
    """
    if strict:
        if np.all(x >= ub):
            violated_indices = (x >= ub).nonzero()
            raise ValueError(
                f"All values of {name} must be > {ub}. Violated at indices {violated_indices}."
            )
    else:
        if np.all(x > ub):
            violated_indices = (x > ub).nonzero()
            raise ValueError(
                f"All values of {name} must be >= {ub}. Violated at indices {violated_indices}."
            )
