import numpy as np
import pandas as pd

from auto_robot_design.motion_planning.dataset_generator import Dataset


def calc_n_sort_df_with_ws(dataset: Dataset) -> pd.DataFrame:
    upd_df = dataset.df.assign(
        total_ws=lambda x: np.sum(
            x.values[
                :, dataset.params_size : dataset.params_size + dataset.ws_grid_size
            ],
            axis=1,
        )
    )
    sorted_df = upd_df.sort_values("total_ws", ascending=False)

    return sorted_df


def filtered_df_with_ws(df: pd.DataFrame, min_ws: int) -> pd.DataFrame:
    return df[df["total_ws"] >= min_ws]


def filtered_csv_dataset(dirpath, max_chunksize, min_ws):
    dataset = Dataset(dirpath)
    path_to_csv_non_filt = dataset.path / "dataset_0.csv"
    path_to_csv_filt = dataset.path / "dataset_filt.csv"
    for chunk in pd.read_csv(path_to_csv_non_filt, chunksize=max_chunksize):
        dataset.df = chunk
        sorted_df = calc_n_sort_df_with_ws(dataset)
        filt_df = filtered_df_with_ws(sorted_df, min_ws)
        if path_to_csv_filt.exists():
            filt_df.to_csv(path_to_csv_filt, mode="a", index_label=False, header=False)
        else:
            filt_df.to_csv(path_to_csv_filt, mode="w", index_label=False)


if __name__ == "__main__":

    dirpath = "/var/home/yefim-work/Documents/auto-robotics-design/top_8"
    filtered_csv_dataset(dirpath, 1e5, 1600)
