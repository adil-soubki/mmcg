#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""An example script"""
from src.core.context import Context
from src.core.app import harness, slurmify


def main(ctx: Context) -> None:
    ctx.parser.add_argument("--foo", action="store_true")
    ctx.parser.set_defaults(modules=["shared"])
    args = slurmify(ctx.parser)
    ctx.log.info("example")


if __name__ == "__main__":
    harness(main)
