# -*- coding: utf-8 -*
import os


def dirparent(path: str, n: int = 1) -> str:
    """
    Calls `os.path.dirname` on the given path n times.

    Args:
        path (str): A file path.
        n (int): The number of parents up to return.

    Examples:
        >>> path = "/example/path/to/something"
        >>> dirparent(path)
        <<< '/example/path/to'
        >>> dirparent(path, 2)
        <<< '/example/path'
    """
    for _ in range(n):
        path = os.path.dirname(path)
    return path
