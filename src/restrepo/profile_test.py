import numpy as np

# import numba.cuda as cuda

from gpu import CRUs, RyR_stationary, LCC_stationary
from params import RestrepoParams


if __name__ == "__main__":

    ci = 0.1 * np.ones((200, 200), dtype=np.float32)
    cs = 0.1 * np.ones((200, 200), dtype=np.float32)
    cp = 0.1 * np.ones((200, 200), dtype=np.float32)

    cjsr = 750.0 * np.ones((200, 200), dtype=np.float32)
    cnsr = 750.0 * np.ones((200, 200), dtype=np.float32)

    CaTi = 20.0 * np.ones((200, 200), dtype=np.float32)
    CaTs = 20.0 * np.ones((200, 200), dtype=np.float32)

    params = RestrepoParams(Ku=1.4e-4, h=2.5)
    RyR = RyR_stationary(cp, params)
    LCC = LCC_stationary(cp, 0.1, params)

    cru_obj = CRUs(params, RyR, LCC, ci, cnsr, cjsr, cs, cp, CaTi, CaTs)
    cru_obj.forward(dt=1e-3, nstep=10000, _tpb2d=8, _tpb1d=16, profile=True)

    print(cru_obj.ICa.copy_to_host().sum())
