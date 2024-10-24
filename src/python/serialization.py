from typing import Any, List
import pickle as pkl

from .model import GWSolution


def save_sim_list(sims: List[Any], fname: str) -> None:
    """Save a list of model simulations

    Args:
        sims (List[Any]): List of GWSolution objects
        fname (str): File name
    """
    if not isinstance(sims, list):
        raise TypeError("sims must be a list")
    sims_as_dicts = [s.to_dict() for s in sims]
    with open(fname, "wb") as f:
        pkl.dump(sims_as_dicts, f)


def load_sim_list(fname: str) -> List[Any]:
    """Load a list of model simulations
    Args:
        fname (str): file name

    Returns:
        List[Any]: List of GWSolution objects
    """
    with open(fname, "rb") as f:
        dicts = pkl.load(f)
    return [GWSolution.from_dict(s) for s in dicts]
