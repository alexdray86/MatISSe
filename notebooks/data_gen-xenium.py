# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: hybiss
#     language: python
#     name: hybiss
# ---

# +
import matplotlib.pyplot as plt
import numpy as np

import tifffile
import pandas as pd

import zarr

from pathlib import Path
# %matplotlib widget
# -

data_root = Path.home()/"data"/"xenium"/"xenium_prerelease_mBrain_large"/"mBrain_ff"
pixel_size = 0.2125

dapi = tifffile.imread(data_root/"morphology_mip.ome.tiff")
dapi.shape

cell_seg = zarr.open(data_root/'cell_segmentation.zarr.zip')
print(cell_seg.tree())

transcripts = pd.read_csv(data_root/"transcript_info.csv.gz")

transcripts["y_location_pixel"] = np.round(transcripts.y_location/pixel_size).astype(int)
transcripts["x_location_pixel"] = np.round(transcripts.x_location/pixel_size).astype(int)


def gene_filter(s):
    return s not in ["Actb+Malat1", "Scgb1a1_129", "Zika_PR_NS5", "Vim_sense"] and not s.startswith("ERCC-") and not s.startswith("DUMMY_")


transcripts = transcripts[transcripts.feature_name.apply(gene_filter)]

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


