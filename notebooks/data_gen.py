# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import pandas as pd
# %matplotlib widget

# Data comes from the following path in the CBAIDT WS: `/data/Toni/HybISS_Raw_Data/e13_5_GLM171wt_e10_#25/E13_5_wt171_10s_25_Crops/Crop3`

dapi = tifffile.imread("../data/C1.tif") # cycle 1 dapi
df = pd.read_csv("../data/wt171_10S_25_Crop3_5.csv")

# # Crop 1

y_idx, x_idx = 6618, 9009
y_idx_2, x_idx_2 = 8263, 12138

df_crop = df[df.y.between(y_idx, y_idx_2) & df.x.between(x_idx, x_idx_2)][["name", "y", "x"]].reset_index(drop=True)
df_crop["y"] = df_crop.y-y_idx
df_crop["x"] = df_crop.x-x_idx

plt.close()
plt.ion()
plt.imshow(dapi[y_idx:y_idx_2,x_idx:x_idx_2], cmap="gray")
plt.scatter(x=df_crop.x, y=df_crop.y, s=1)

df_crop.to_csv("../data/crop1_genes.csv", index=False)
tifffile.imwrite("../data/crop1_dapi.tif", dapi[y_idx:y_idx_2,x_idx:x_idx_2])

# # Crop 2

y_idx, x_idx = 12793, 11701
y_idx_2, x_idx_2 = 14000, 14000

df_crop_2 = df[df.y.between(y_idx, y_idx_2) & df.x.between(x_idx, x_idx_2)][["y", "x"]].reset_index(drop=True)
df_crop_2["y"] = df_crop_2.y-y_idx
df_crop_2["x"] = df_crop_2.x-x_idx

plt.close()
plt.ion()
plt.imshow(dapi[y_idx:y_idx_2,x_idx:x_idx_2], cmap="gray")
plt.scatter(x=df_crop_2.x, y=df_crop_2.y, s=1)

df_crop_2.to_csv("../data/crop2_genes.csv", index=False)
tifffile.imwrite("../data/crop2_dapi.tif", dapi[y_idx:y_idx_2,x_idx:x_idx_2])


