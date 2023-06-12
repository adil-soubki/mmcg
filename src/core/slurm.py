# -*- coding: utf-8 -*-
import io
import logging
import os
import subprocess
import sys
from typing import Dict, Iterable, Optional

import pandas as pd

from .functional import safe_iter


_FLAGS = {
    "job-name": os.path.basename(sys.argv[0]),
    "output": "/home/%u/scratch/logs/%x.%j.out",
    "partition": "defq",
    "time": "08:00:00",
}


def sbatch(
    cmds: Iterable[str],
    flags: Optional[Dict[str, str]] = None,
    modules: Optional[Iterable[str]] = None,
    dryrun: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    # Parse inputs.
    cmds = safe_iter(cmds)
    flags = flags or _FLAGS
    for key in _FLAGS:
        if key not in flags:
            flags[key] = _FLAGS[key]
    modules = modules or []
    # Prepare batch script.
    stdin = ["#!/bin/bash"]
    for key, val in flags.items():
        stdin.append(f"#SBATCH --{key}={val}")
    if modules:
        stdin.append("")
    for module in modules:
        stdin.append(f"module load {module}")
    stdin.append("")
    stdin += cmds
    log = logging.getLogger(__name__)
    # Handle dryruns.
    if dryrun:
        stdin.append("=" * 32 + " DRYRUN " + "=" * 32)
        stdin.insert(0, "=" * 32 + " DRYRUN " + "=" * 32)
        log.info("Would submit to slurm.\n%s", "\n".join(stdin))
        return subprocess.CompletedProcess("", 0)
    log.info("Submitting to slurm.\n%s", "\n".join(stdin))
    # Submit the job.
    return subprocess.run(
        ["sbatch"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        input="\n".join(stdin).encode("utf-8"),
        check=True,
    )


def sinfo() -> pd.DataFrame:
    cproc = subprocess.run("sinfo", capture_output=True, check=True)
    return pd.read_csv(io.StringIO(cproc.stdout.decode("utf-8")), sep="\s+")


def timelimit(partition: str) -> str:
    df = sinfo()
    if partition not in df.PARTITION.unique():
        raise ValueError(f"unknown partition: {partition}")
    tls = df[df.PARTITION == partition].TIMELIMIT.unique()
    assert len(tls) == 1
    return str(tls[0])
