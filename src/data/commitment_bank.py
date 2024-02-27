# -*- coding: utf-8 -*-
import os

import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..core.path import dirparent


CB_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "cb")
CB_TO_7_CLS = {v: k for k, v in enumerate(range(-3, 4))}
CB_TO_5_CLS = {
    -3: 0,
    -2: 1,
    -1: 1,
    0: 2,
    1: 3,
    2: 3,
    3: 4,
}


def load(num_labels: int) -> datasets.Dataset:
    assert num_labels in (5, 7)
    df = pd.read_json(os.path.join(CB_DIR, "annotations.jsonl"), lines=True)
    # Load opensmile features.
    df = df.merge(load_opensmile(), on="audio_file", validate="1:1")
    ftrs = df["opensmile_features"]
    # Add full audio path.
    df = df.assign(
        audio=os.path.join(CB_DIR, "audio") + os.sep + df.audio_file
    )[["number", "clip_start", "clip_end", "audio", "cb_target", "cb_val"]].astype(str)
    # Some rows (exactly 1) do not have a switchboard file to match so number is blank.
    df = df.replace("", np.nan).dropna().astype({"number": int, "cb_val": float})
    # XXX: Convert to 5 label classification for now.
    df = df.assign(cb_val_float=df.cb_val)
    if num_labels == 5:
        df = df.assign(cb_val=df.cb_val.round().astype(int).map(lambda v: CB_TO_5_CLS[v]))
    elif num_labels == 7:
        df = df.assign(cb_val=df.cb_val.round().astype(int).map(lambda v: CB_TO_7_CLS[v]))
    # Create dataset.
    cb = datasets.Dataset.from_pandas(df, preserve_index=False)
    cb = cb.cast_column("audio", datasets.Audio(sampling_rate=16_000))
    cb = cb.add_column("opensmile_features", ftrs)
    return cb


def load_kfold(
    num_labels: int, fold: int, k: int = 5, seed: int = 42
) -> datasets.DatasetDict:
    assert fold >= 0 and fold <= k - 1
    kf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=True)
    cb = load(num_labels)
    train_idxs, test_idxs = list(kf.split(cb, cb["cb_val"]))[fold]
    return datasets.DatasetDict({
        "train": cb.select(train_idxs),
        "test": cb.select(test_idxs),
    })


def load_opensmile() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(CB_DIR, "opensmile.csv"))
    df = df.assign(
        audio_file=df.file.str.replace("_features", "").str.replace(".csv", ".wav")
    )
    fcols = [c for c in df.columns if c.isnumeric()]
    features = df[fcols].to_numpy()
    return pd.DataFrame({
        "opensmile_features": list(features),
        "audio_file": df.audio_file
    })
