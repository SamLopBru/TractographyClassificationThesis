import nibabel as nib
import numpy as np
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import length as sl_length
import pathlib
import os

# DUMMY_PATH = "/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives/testset/sub-1019/anat/sub-1019__T1w.nii.gz"
# TRACTOGRAPHY_PATH = "/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives/testset/sub-1019/tractography/"

# TRACT_NAMES = [
#     'AF_L','AF_R','CC_Fr_1','CC_Fr_2','CC_Oc','CC_Pa','CC_Pr_Po','CG_L','CG_R',
#     'FAT_L','FAT_R','FPT_L','FPT_R','FX_L','FX_R','IFOF_L','IFOF_R','ILF_L','ILF_R',
#     'MCP','MdLF_L','MdLF_R','OR_ML_L','OR_ML_R','POPT_L','POPT_R','PYT_L','PYT_R',
#     'SLF_L','SLF_R','UF_L','UF_R'
# ]

# def _init_dict():
#     return {k: [] for k in TRACT_NAMES}

# TRACTS_START = _init_dict()
# TRACTS_END = _init_dict()

# TRACTS_ARCLEN = _init_dict()          # per-streamline arc length
# TRACTS_CHORD = _init_dict()           # per-streamline endpoint distance
# TRACTS_TORT = _init_dict()            # per-streamline tortuosity = arclen/chord
# TRACTS_MAX_DEV = _init_dict()         # max distance to chord line
# TRACTS_MEAN_DEV = _init_dict()        # mean distance to chord line
# TRACTS_MEAN_CURV = _init_dict()       # mean curvature proxy
# TRACTS_MAX_CURV = _init_dict()        # max curvature proxy

# def _name_tract(tract: str) -> str:
#     return tract.split("__")[-1].split(".")[0]

# def _point_line_distance(P, A, B, eps=1e-8):
#     AB = B - A
#     denom = np.linalg.norm(AB) + eps
#     AP = P - A
#     cross = np.cross(AP, AB)
#     return np.linalg.norm(cross, axis=1) / denom

# def _curvature_proxy(points, eps=1e-8):
#     if points.shape[0] < 3:
#         return 0.0, 0.0
#     v = points[1:] - points[:-1]                       # (N-1,3)
#     vn = np.linalg.norm(v, axis=1, keepdims=True) + eps
#     t = v / vn                                         # unit tangents (N-1,3)
#     dt = t[1:] - t[:-1]                                # (N-2,3)
#     ds = (vn[1:] + vn[:-1])[:, 0] / 2.0                # approx arc-length step
#     kappa = np.linalg.norm(dt, axis=1) / (ds + eps)    # curvature proxy
#     return float(np.mean(kappa)), float(np.max(kappa))

# for fname in os.listdir(TRACTOGRAPHY_PATH):
#     if not fname.endswith(".trk"):
#         continue

#     tract_name = _name_tract(fname)
#     if tract_name not in TRACTS_START:
#         continue

#     sft = load_tractogram(str(pathlib.Path(TRACTOGRAPHY_PATH) / fname), str(DUMMY_PATH))
#     streamlines = sft.streamlines

#     for sl in streamlines:
#         sl = np.asarray(sl, dtype=np.float32)
#         if sl.shape[0] < 2:
#             continue

#         A = sl[0]
#         B = sl[-1]

#         TRACTS_START[tract_name].append(A)
#         TRACTS_END[tract_name].append(B)

#         arclen = float(sl_length(sl))
#         chord = float(np.linalg.norm(B - A))
#         tort = arclen / (chord + 1e-8)

#         d = _point_line_distance(sl, A, B)
#         mean_dev = float(np.mean(d))
#         max_dev = float(np.max(d))

#         mean_curv, max_curv = _curvature_proxy(sl)

#         TRACTS_ARCLEN[tract_name].append(arclen)
#         TRACTS_CHORD[tract_name].append(chord)
#         TRACTS_TORT[tract_name].append(tort)
#         TRACTS_MEAN_DEV[tract_name].append(mean_dev)
#         TRACTS_MAX_DEV[tract_name].append(max_dev)
#         TRACTS_MEAN_CURV[tract_name].append(mean_curv)
#         TRACTS_MAX_CURV[tract_name].append(max_curv)

# def _summ(x):
#     x = np.asarray(x, dtype=np.float32)
#     if x.size == 0:
#         return (np.nan, np.nan, np.nan, np.nan)
#     return (float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x)))

# for tract in TRACT_NAMES:
#     start_pts = np.asarray(TRACTS_START[tract], dtype=np.float32)
#     end_pts = np.asarray(TRACTS_END[tract], dtype=np.float32)

#     start_mean = np.mean(start_pts, axis=0) if start_pts.size else np.array([np.nan]*3, dtype=np.float32)
#     end_mean = np.mean(end_pts, axis=0) if end_pts.size else np.array([np.nan]*3, dtype=np.float32)

#     arclen_mu, arclen_sd, arclen_min, arclen_max = _summ(TRACTS_ARCLEN[tract])
#     chord_mu, chord_sd, chord_min, chord_max = _summ(TRACTS_CHORD[tract])
#     tort_mu, tort_sd, tort_min, tort_max = _summ(TRACTS_TORT[tract])
#     mdev_mu, mdev_sd, mdev_min, mdev_max = _summ(TRACTS_MEAN_DEV[tract])
#     xdev_mu, xdev_sd, xdev_min, xdev_max = _summ(TRACTS_MAX_DEV[tract])
#     mcurv_mu, mcurv_sd, mcurv_min, mcurv_max = _summ(TRACTS_MEAN_CURV[tract])
#     xcurv_mu, xcurv_sd, xcurv_min, xcurv_max = _summ(TRACTS_MAX_CURV[tract])

#     print(f"\nTract {tract}:")
#     print(f"  Start mean: {start_mean}, End mean: {end_mean}")
#     print(f"  Arc length: mean={arclen_mu:.3f} ± {arclen_sd:.3f} (min={arclen_min:.3f}, max={arclen_max:.3f})")
#     print(f"  Chord dist: mean={chord_mu:.3f} ± {chord_sd:.3f} (min={chord_min:.3f}, max={chord_max:.3f})")
#     print(f"  Tortuosity: mean={tort_mu:.3f} ± {tort_sd:.3f} (min={tort_min:.3f}, max={tort_max:.3f})")
#     print(f"  Mean dev  : mean={mdev_mu:.3f} ± {mdev_sd:.3f} (min={mdev_min:.3f}, max={mdev_max:.3f})")
#     print(f"  Max dev   : mean={xdev_mu:.3f} ± {xdev_sd:.3f} (min={xdev_min:.3f}, max={xdev_max:.3f})")
#     print(f"  Mean curv : mean={mcurv_mu:.5f} ± {mcurv_sd:.5f} (min={mcurv_min:.5f}, max={mcurv_max:.5f})")
#     print(f"  Max curv  : mean={xcurv_mu:.5f} ± {xcurv_sd:.5f} (min={xcurv_min:.5f}, max={xcurv_max:.5f})")


import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils.dataloader import StreamlineDataset

import numpy as np

train_dir = Path("sequences/trainset")
train_files = sorted([str(f) for f in train_dir.glob("*.hdf5")])
print(f"Found {len(train_files)} training HDF5 files")

train_dataset = StreamlineDataset(
    train_files,
    sampling_percentage=1.0,
    max_streamlines_per_tract=None,
    full_sample_threshold=1000
)


import pandas as pd
import plotly.express as px
import numpy as np
unique, counts = np.unique(train_dataset.streamline_index['tract_id'], return_counts=True)
# for bundle_id, count in zip(unique, counts):
#     print(f"  Bundle {bundle_id}: {count} samples") 


df = pd.DataFrame({"bundle_id": unique.astype(int), "samples": counts.astype(int)})

fig = px.bar(
    df,
    x="bundle_id",
    y="samples",
    text="samples",
    title="Samples per bundle (trainset)"
)
fig.update_traces(textposition="outside", cliponaxis=False)
fig.update_xaxes(title_text="Bundle ID", type="category")  # keep categories, not numeric axis
fig.update_yaxes(title_text="Samples")

fig.show()
