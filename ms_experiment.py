import os
import sys

sys.path.append(os.path.abspath(os.getcwd()) + "/src")

import argparse

import numpy as np
import h5py

from src import GWParameters, GWModel, save_sim_group, create_log, log_args, log_dict

# TODO: Make the size of the files much, much smaller. 17 experiments creates a 2.2Gb folder!
# Some ideas: Collect at larger intervals, convert to float32, convert LCC_i and ClCh to bools, enumerate RyR state space and save as int, use int 16, don't save parameters more than once


def main(args):
    logger = create_log(args.fname + ".log", path=os.getcwd())
    log_args(logger, args, indent=True)

    XKr = np.array([0.999503, 4.13720e-4, 7.27568e-5, 8.73984e-6, 1.36159e-6])
    XKv14 = np.array(
        [
            0.953060,
            0.0253906,
            2.53848e-4,
            1.12796e-6,
            1.87950e-9,
            0.0151370,
            0.00517622,
            8.96600e-4,
            8.17569e-5,
            2.24032e-6,
        ]
    )
    XKv43 = np.array(
        [
            0.722328,
            0.101971,
            0.00539932,
            1.27081e-4,
            1.82742e-6,
            0.152769,
            0.00962328,
            0.00439043,
            0.00195348,
            0.00143629,
        ]
    )

    CaSS = np.ones((args.NCaRU, 4)) * 1.45370e-4
    CaJSR = np.ones(args.NCaRU) * 0.908408
    LCC = np.random.choice([1, 2, 7], p=[0.958, 0.038, 0.004], size=(args.NCaRU, 4))
    LCC_inactivation = np.random.choice(
        [0, 1], p=[0.0575, 0.9425], size=(args.NCaRU, 4)
    )
    ClCh = np.random.choice([0, 1], p=[0.998, 0.002], size=(args.NCaRU, 4))
    RyR = np.zeros((args.NCaRU, 4, 6), dtype=np.int32)
    for i in range(args.NCaRU):
        for j in range(4):
            for _ in range(5):
                idx = np.random.choice([0, 4, 5], p=[0.609, 0.5 * 0.391, 0.5 * 0.391])
                RyR[i, j, idx] += 1

    init_state = {
        "V": -91.382,
        "Nai": 10.0,
        "Ki": 131.84,
        "Cai": 1.45273e-4,
        "CaNSR": 0.908882,
        "CaLTRPN": 8.9282e-3,
        "CaHTRPN": 0.137617,
        "m": 5.33837e-4,
        "h": 0.996345,
        "j": 0.997315,
        "xKs": 2.04171e-4,
        "XKr": XKr,
        "XKv14": XKv14,
        "XKv43": XKv43,
        "CaSS": CaSS,
        "CaJSR": CaJSR,
        "LCC": LCC,
        "LCC_inactivation": LCC_inactivation,
        "RyR": RyR,
        "ClCh": ClCh,
    }
    Istim = lambda t: 35 if t < 2 else 0  # Stimulus function

    params = GWParameters(
        NCaRU_sim=args.NCaRU
    )  # Create parameters object using default values
    logger.info("Parameters:", extra={"simple": True})
    log_dict(logger, params.to_dict(), indent=True)

    model = GWModel(parameters=params, stimulus_fn=Istim, init_state=init_state)

    direc = "experiments"
    if not os.path.exists(direc):
        os.makedirs(direc)

    with h5py.File(direc + "/" + args.fname + ".h5py", "w") as f:
        par_grp = f.create_group("params")
        for par_name, par_val in params.to_dict().items():
            par_grp.create_dataset(name=par_name, data=par_val)
        del par_grp

        for i in range(args.nsim):
            sim = model.simulate(
                step_size=args.step_size,
                num_steps=args.num_steps,
                record_every=args.record_every,
            )
            grp_name = f"sim{i}"
            save_sim_group(sim, f, grp_name)
            del sim

    # all_sims = [
    #    model.simulate(
    #        step_size=args.step_size,
    #        num_steps=args.num_steps,
    #        record_every=args.record_every,
    #    )
    #    for _ in range(args.nsim)
    # ]
    # save_sim_list(all_sims, args.fname + ".pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str, help="filename")
    parser.add_argument("nsim", type=int, help="Number of times to simulate")
    parser.add_argument(
        "--NCaRU", type=int, help="Number of CaRUs to simulate", default=1000
    )
    parser.add_argument(
        "--step-size", type=float, help="Euler time step size", default=1e-3
    )
    parser.add_argument(
        "--num-steps", type=int, help="Number of Euler steps", default=500_000
    )
    parser.add_argument(
        "--record-every",
        type=int,
        help="Number of Euler steps between recording of states",
        default=1000,
    )
    args = parser.parse_args()
    main(args)
