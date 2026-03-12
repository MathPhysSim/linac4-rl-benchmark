"""Plot initial beam states from HDF5 measurement data.

Loads the initial-state snapshot from ``init_states.h5`` and plots
BPM/BCT-related and corrector-related columns separately.
"""

import pandas as pd
import matplotlib.pyplot as plt

frame = pd.read_hdf("init_states.h5", key="init")

bct_cols = [col for col in frame.columns if "BCT" in col or "BPUSE" in col]
other_cols = [col for col in frame.columns if "BCT" not in col and "BPUSE" not in col]

frame.loc[:, bct_cols].T.plot()
plt.title("BCT / BPUSE columns")
plt.show()

frame.loc[:, other_cols].T.plot()
plt.title("Other columns")
plt.show()