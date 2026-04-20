import scipy.io as sio
import numpy as np

SCENARIO_PATH = r'C:\Users\Lenovo\PycharmProjects\QuantRIC\deepmimo_scenarios\o1_28b'

def load_deepmimo_scenario(num_samples=5):
    pl_raw  = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.PL.mat')
    loc_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.Loc.mat')
    los_raw = sio.loadmat(f'{SCENARIO_PATH}\\O1_28B.1.LoS.mat')

    pl_data  = pl_raw['PL_array_full']         # (497931, 2)
    loc_data = loc_raw['Loc_array_full']        # (497931, 6)
    los_data = los_raw['LOS_tag_array_full']    # check shape next

    samples = []
    for i in range(num_samples):
        sample = {
            'index'    : i,
            'path_loss': pl_data[i, 0],
            'location' : loc_data[i, :3],
            'los'      : los_data[0,i],
        }
        samples.append(sample)

    return samples