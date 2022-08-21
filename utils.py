import os
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
import skimage.measure
import tensorflow as tf
import torch
from ivis import Ivis
from loguru import logger
from scipy.sparse import csr_matrix, vstack
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from tqdm.notebook import tqdm


def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_h5(filename, batch_size=10000):
    start, stop = 0, batch_size
    df = pd.read_hdf(filename, start=start, stop=stop)
    index = 0
    while len(df) > 0:
        print(". ", end="", flush=True)
        if (index + 1) % 30 == 0:
            print(flush=True)
        yield df
        start, stop = stop, stop + batch_size
        df = pd.read_hdf(filename, start=start, stop=stop)
        index += 1


def read_cell_ids(
    filename,
    df_evaluation,
    batch_size=10000,
    reduce_dimension=False,
    reduce_by=10,
    reduce_function="max",
    only_exist_in_eval=False,
):
    eval_cell_ids = set(df_evaluation.cell_id.values)
    reduce_function = getattr(np, reduce_function)
    df_cell_ids, data = [], []
    for df in read_h5(filename, batch_size=batch_size):
        if only_exist_in_eval:
            df = df[df.index.isin(eval_cell_ids)]
        df_cell_ids.extend(df.index.to_list())
        if reduce_dimension:
            v = df.to_numpy()
            embed = skimage.measure.block_reduce(v, (1, reduce_by), reduce_function)
            data.append(embed)

    df_cell_ids = pd.DataFrame(df_cell_ids, columns=["cell_id"])

    if reduce_dimension:
        data = np.concatenate(data)
        return df_cell_ids, data
    return df_cell_ids, None


def train_ivis(
    filename_or_data,
    meta_df,
    supervised=True,
    ivis_dim=2,
    epochs=10,
    k=150,
    build_index_on_disk=True,
    precompute=True,
    n_trees=50,
):
    if supervised:
        Y = pd.Categorical(meta_df["cell_type"]).codes
    else:
        Y = None
    model = Ivis(
        embedding_dims=ivis_dim,
        epochs=epochs,
        k=k,
        build_index_on_disk=build_index_on_disk,
        precompute=precompute,
        n_trees=n_trees,
    )
    if isinstance(filename_or_data, Path):
        with h5py.File(filename_or_data, "r") as f:
            X = f[filename_or_data.stem]["block0_values"]
            embeddings = model.fit_transform(
                X, Y, shuffle_mode="batch"
            )  # Shuffle batches when using h5 files
    else:
        embeddings = model.fit_transform(filename_or_data, Y, shuffle_mode="batch")

    return model, embeddings


def transform(
    filename,
    model,
    df_evaluation,
    reduce_dimension=False,
    reduce_by=10,
    reduce_function="max",
    batch_size=1000,
):
    reduce_function = getattr(np, reduce_function)
    eval_cell_ids = set(df_evaluation.cell_id.values)
    data = []
    for df in read_h5(filename, batch_size=batch_size):
        df = df[df.index.isin(eval_cell_ids)]
        v = df.to_numpy()
        if reduce_dimension:
            v = skimage.measure.block_reduce(v, (1, reduce_by), reduce_function)
        embed = model.transform(v)
        data.append(embed)
    embeddings = np.concatenate(data)
    return embeddings


def find_knn(train, test, n_neighbors=1000, radius=0.1):
    neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=radius)
    neigh.fit(train)
    kneighbors = neigh.kneighbors(test, return_distance=False)
    return kneighbors


def read_hd5_sparse(filename, batch_size=1000):
    columns = None
    data = []
    for df in tqdm(read_h5(filename, batch_size=batch_size)):
        if columns is None:
            columns = df.columns
        v = csr_matrix(df.to_numpy())
        data.append(v)
    data = vstack(data)
    return data, columns


def calculate_target(filename, kneighbors):
    target_values, target_columns = read_hd5_sparse(filename)
    output_targets = []
    for i, inds in tqdm(enumerate(kneighbors)):
        v = target_values[inds]
        v_mean = np.mean(v, axis=0)
        output_targets.append(v_mean)
    output_targets = np.stack(output_targets)
    output_targets_df = pd.DataFrame(output_targets, columns=target_columns)
    return output_targets_df

