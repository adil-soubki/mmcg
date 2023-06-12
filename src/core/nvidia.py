# -*- coding: utf-8 -*-
import io
import re
import subprocess
from typing import Any

import pandas as pd


def query_gpu_props() -> list[str]:
    ret = []
    lines = (
        subprocess.run(
            ["nvidia-smi", "--help-query-gpu"], capture_output=True, check=True
        )
        .stdout.decode("utf-8")
        .split("\n")
    )

    for line in lines:
        match = re.match(r'^"(\S+)"', line)
        if match and not match.group(1)[0].isupper():
            ret.append(match.group(1))
    return ret


def query_gpu() -> pd.DataFrame:
    props = [
        "timestamp",
        "index",
        "gpu_name",
        "temperature.gpu",
        "utilization.gpu",
        "utilization.memory",
        "memory.total",
        "memory.free",
        "memory.used",
    ]
    cmd = ["nvidia-smi", f"--query-gpu={','.join(props)}", "--format=csv"]
    cproc = subprocess.run(cmd, capture_output=True, check=True)

    def process(row: "pd.Series[Any]") -> "pd.Series[Any]":
        for idx in row.index:
            if "MiB" in idx:
                row[idx] = int(row[idx].replace("MiB", "").strip())
            if "%" in idx:
                row[idx] = int(row[idx].replace("%", "").strip())
        return row

    df = pd.read_csv(io.StringIO(cproc.stdout.decode("utf-8")))
    df.columns = pd.Index([cname.strip() for cname in df.columns])
    return df.apply(process, axis=1)


def best_gpu() -> int:
    return int(
        query_gpu().sort_values("memory.free [MiB]", ascending=False).iloc[0]["index"]
    )
