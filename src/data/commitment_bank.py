# -*- coding: utf-8 -*-
import os

import datasets
import numpy as np
import pandas as pd

from ..core.path import dirparent


CB_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "cb")
CB_TO_5_CLS = {
    -3: 0,
    -2: 1,
    -1: 1,
    0: 2,
    1: 3,
    2: 3,
    3: 4,
}


def load() -> datasets.Dataset:
    df = pd.read_json(os.path.join(CB_DIR, "annotations.jsonl"), lines=True)
    df = df.assign(
        audio=os.path.join(CB_DIR, "audio") + os.sep + df.audio_file
    )[["number", "clip_start", "clip_end", "audio", "cb_target", "cb_val"]].astype(str)
    # Some rows (exactly 1) do not have a switchboard file to match so number is blank.
    df = df.replace("", np.nan).dropna().astype({"number": int, "cb_val": float})
    # XXX: Convert to 5 label classification for now.
    df = df.assign(cb_val=df.cb_val.round().astype(int).map(lambda v: CB_TO_5_CLS[v]))
    cb = datasets.Dataset.from_pandas(df, preserve_index=False)
    cb = cb.cast_column("audio", datasets.Audio(sampling_rate=16_000))
    return cb
