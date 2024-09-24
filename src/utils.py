from typing import Any
import json

import numpy as np
import numpy.typing as npt


def RyR_state_to_int16(RyRs: npt.NDArray) -> npt.NDArray:
    # If 3 and 4 are in equilibrium, then the way I wrote the code makes -ve values possible, so get rid of these
    RyRs_34 = RyRs[..., 2] + RyRs[..., 3]
    RyR_3or4_neg = np.logical_or(RyRs[..., 2] < 0, RyRs[..., 3] < 0)
    RyRs2 = np.logical_not(RyR_3or4_neg) * RyRs[..., 2] + RyR_3or4_neg * RyRs_34
    RyRs3 = np.logical_not(RyR_3or4_neg) * RyRs[..., 3]

    # Same as above with 5 and 6
    RyRs_56 = RyRs[..., 4] + RyRs[..., 5]
    RyR_5or6_neg = np.logical_or(RyRs[..., 4] < 0, RyRs[..., 5] < 0)
    RyRs4 = np.logical_not(RyR_5or6_neg) * RyRs[..., 4] + RyR_5or6_neg * RyRs_56
    # RyRs5 = np.logical_not(RyR_5or6_neg) * RyRs[..., 5]

    out = RyRs[..., 0] + 6 * RyRs[..., 1] + 36 * RyRs2 + 216 * RyRs3 + 1296 * RyRs4
    return np.int16(out)


def load_model_from_json(fname: str) -> dict:
    """Load model related data from fname.json into a dictionary."""
    with open(fname + ".json", "r", encoding="utf-8") as f:
        model_data = json.load(f)
    return model_data


def create_boilerplate(model_data: dict) -> str:
    """Generate the boilerplate for parameters class fields."""
    out_str = ""
    for name, info in model_data["parameters"].items():
        if info["constraint"] == "positive":
            constraint_fn = "assert_positive"
            constraint_symbol = "> 0"
        elif info["constraint"] == "non-negative":
            constraint_fn = "assert_nonnegative"
            constraint_symbol = ">= 0"
        out_str += f"""
    @property
    def {name}(self) -> float:
        '''float: {info["description"]}. Value must be {constraint_symbol}. Defaults to {info["default"]}.
        '''
        return self.__cxx_struct.{name}

    @{name}.setter
    def {name}(self, value: float) -> None:
        {constraint_fn}(value, "{name}")
        self.__cxx_struct.{name} = value
    """
    return out_str


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
    x: npt.NDArray, lb: np.ScalarType, name: str, strict: bool = False
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
    x: npt.NDArray, ub: np.ScalarType, name: str, strict: bool = False
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
