from typing import Any, List, Dict
import pickle as pkl

import numpy as np
import h5py

from .model import GWSolution
from .utils import RyR_state_to_int16


def sim_dict_reduced_memory(sim: Any) -> Dict:
    sim_dict = sim.to_dict()["vars"]
    # Convert floats to float 32
    sim_dict["t"] = np.float32(sim_dict["t"])
    sim_dict["V"] = np.float32(sim_dict["V"])
    sim_dict["Nai"] = np.float32(sim_dict["Nai"])
    sim_dict["Ki"] = np.float32(sim_dict["Ki"])
    sim_dict["Cai"] = np.float32(sim_dict["Cai"])
    sim_dict["CaNSR"] = np.float32(sim_dict["CaNSR"])
    sim_dict["CaLTRPN"] = np.float32(sim_dict["CaLTRPN"])
    sim_dict["CaHTRPN"] = np.float32(sim_dict["CaHTRPN"])
    sim_dict["m"] = np.float32(sim_dict["m"])
    sim_dict["h"] = np.float32(sim_dict["h"])
    sim_dict["j"] = np.float32(sim_dict["j"])
    sim_dict["xKs"] = np.float32(sim_dict["xKs"])
    sim_dict["XKr"] = np.float32(sim_dict["XKr"][:, :4])
    sim_dict["XKv14"] = np.float32(sim_dict["XKv14"][:, :9])
    sim_dict["XKv43"] = np.float32(sim_dict["XKv43"][:, :9])
    sim_dict["CaSS"] = np.float32(sim_dict["CaSS"])
    sim_dict["CaJSR"] = np.float32(sim_dict["CaJSR"])

    # Convert LCC to int8 and the binary varaibles to bools
    sim_dict["LCC"] = np.int8(sim_dict["LCC"])
    sim_dict["LCC_inactivation"] = np.bool_(sim_dict["LCC_inactivation"])
    sim_dict["ClCh"] = np.bool_(sim_dict["ClCh"])

    # Encode RyR state via string
    sim_dict["RyR"] = RyR_state_to_int16(sim_dict["RyR"])
    return sim_dict


def save_sim(sim: Any, fname: str) -> None:
    params = sim.to_dict()["params"]
    vars_ = sim_dict_reduced_memory(sim)

    with h5py.File(fname, "w") as f:
        param_grp = f.create_group("params")
        for par_name, par_val in params.items():
            param_grp.create_dataset(name=par_name, data=par_val, compression="gzip")

        var_grp = f.create_group("vars")
        for var_name, var_val in vars_.items():
            var_grp.create_dataset(name=var_name, data=var_val, compression="gzip")


def save_sim_group(sim: Any, f: h5py.File, group_name: str) -> None:
    """Save a list of model simulations

    Args:
        sims (List[Any]): List of GWSolution objects
        fname (str): File name
    """
    vars_ = sim_dict_reduced_memory(sim)
    grp = f.create_group(group_name)
    for var_name, var_val in vars_.items():
        grp.create_dataset(name=var_name, data=var_val, compression="gzip")


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
