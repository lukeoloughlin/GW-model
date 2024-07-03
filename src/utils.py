from typing import Any, List, Tuple


def create_boilerplate(names: List[Tuple[str, str, str, float]]) -> str:
    """Generate the boilerplate for parameters class fields."""
    out_str = ""
    for name, constraint, description, default in names:
        if constraint == "p":
            constraint_fn = "assert_positive"
            constraint_symbol = "> 0"
        else:
            constraint_fn = "assert_nonnegative"
            constraint_symbol = ">= 0"
        out_str += f"""
    @property
    def {name}(self) -> float:
        '''float: {description}. Value must be {constraint_symbol}. Defaults to {default}.
        '''
        return self.__cxx_struct.{name}

    @{name}.setter
    def {name}(self, value: float) -> None:
        {constraint_fn}(value, "{name}")
        self.__cxx_struct.{name} = value
    """
    return out_str


def add_docstring(docstring: str) -> Any:
    """Decorator to add a docstring to a class

    Args:
        docstring (str): Docstring to apply to class
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
