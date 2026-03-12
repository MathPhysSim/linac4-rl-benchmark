"""Visualise the measured Linac4 response matrix as a heatmap.

Loads the pickled response matrix and plots either the horizontal or
vertical plane using seaborn.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_pickle("linac4_rm.pcl")
df.columns = [entry.split("/")[0] for entry in df.columns]
df.index = [entry.split("/")[0] for entry in df.index]

# Select the vertical plane (rows 17+, columns 16+)
df = df.iloc[17:, 16:]

plt.figure(figsize=(10, 10))
sns.set(font_scale=1.5)
sns.heatmap(df, square=True, cmap="YlGnBu")
plt.title("Vertical response", size=18)
plt.tight_layout()
plt.savefig("Response_matrix_ver.png")
plt.show()
