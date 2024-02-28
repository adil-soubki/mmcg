# -*- coding: utf-8 -*
from typing import Optional

from pandas import DataFrame  # type: ignore


def update(ldf: DataFrame, rdf: DataFrame, on: Optional[list[str]] = None) -> DataFrame:
    assert len(set(ldf.columns) ^ set(rdf.columns)) == 0
    cols = ldf.columns
    if on:
        ldf = ldf.set_index(on)
        rdf = rdf.set_index(on)
    return DataFrame(ldf.to_dict("index") | rdf.to_dict("index")).T.reset_index(
        names=on
    )[cols]
