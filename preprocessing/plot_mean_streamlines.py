import struct
import pathlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.dataset_handler import Tractoinferno_handler

ENCODED_TRACTS: dict[str, int] = {
    'AF_L': 0, 'AF_R': 0, 'CC_Fr_1': 0, 'CC_Fr_2': 0, 'CC_Oc': 0,
    'CC_Pa': 0, 'CC_Pr_Po': 0, 'CG_L': 0, 'CG_R': 0, 'FAT_L': 0,
    'FAT_R': 0, 'FPT_L': 0, 'FPT_R': 0, 'FX_L': 0, 'FX_R': 0,
    'IFOF_L': 0, 'IFOF_R': 0, 'ILF_L': 0, 'ILF_R': 0, 'MCP': 0,
    'MdLF_L': 0, 'MdLF_R': 0, 'OR_ML_L': 0, 'OR_ML_R': 0,
    'POPT_L': 0, 'POPT_R': 0, 'PYT_L': 0, 'PYT_R': 0,
    'SLF_L': 0, 'SLF_R': 0, 'UF_L': 0, 'UF_R': 0
}

def count_streamlines_trk(trk_path: str) -> int:
    with open(trk_path, 'rb') as f:
        f.seek(988)
        return struct.unpack('<i', f.read(4))[0]

def _name_tract(tract: pathlib.Path) -> str:
    return tract.name.split("__")[-1].split(".")[0]

def process_subject(subject_data: dict) -> dict:
    local_counts = {k: 0 for k in ENCODED_TRACTS}
    for tract in subject_data['tracts']:
        trk_path = str(tract)
        name = _name_tract(pathlib.Path(trk_path))
        if name in local_counts:
            local_counts[name] += count_streamlines_trk(trk_path)
    return local_counts

if __name__ == "__main__":
    SPLITS = ['trainset', 'validset', 'testset']
    split_means = {}

    for scope in SPLITS:
        dataset_handler = Tractoinferno_handler(
            '/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives',
            scope=scope
        )
        subjects_data = {s['subject']: s for s in dataset_handler.get_data()}
        n_subjects = len(subjects_data)

        serialized = [
            {'tracts': [str(t) for t in s['tracts']]}
            for s in subjects_data.values()
        ]

        split_counts = {k: 0 for k in ENCODED_TRACTS}

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(process_subject, serialized))

        for result in results:
            for k, v in result.items():
                split_counts[k] += v

        split_means[scope] = {k: v / n_subjects for k, v in split_counts.items()}

    # Grouped bar plot — one group per bundle, one bar per split
    colors = {'trainset': '#1f77b4', 'validset': '#ff7f0e', 'testset': '#2ca02c'}
    bundles = list(ENCODED_TRACTS.keys())

    fig = go.Figure([
        go.Bar(
            name=scope,
            x=bundles,
            y=[split_means[scope][b] for b in bundles],
            marker_color=colors[scope]
        )
        for scope in SPLITS
    ])

    fig.update_layout(
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
    )
    fig.update_xaxes(title_text="Bundle", tickangle=45)
    fig.update_yaxes(
        title_text="Mean Streamlines (log)",
        type="log",
        tickmode="array",
        tickvals=[10, 100, 1000, 10000, 100000],
        ticktext=["10", "100", "1k", "10k", "100k"]
    )
    fig.update_traces(cliponaxis=False)
    fig.write_image("mean_tracts_splits.png")
