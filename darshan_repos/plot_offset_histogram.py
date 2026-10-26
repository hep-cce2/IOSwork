import sys
import argparse
import bisect
from typing import Any, Union
import json

import darshan

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

import pandas as pd
import numpy as np

def format_bytes(y, pos):
    if y == 0: return "0B"
    if not np.isfinite(y): return str(y)
    size_name = ("B", "KB", "MB", "GB", "TB", "PB")
    epsilon = 1e-9
    temp_y_div = y
    div_i = 0
    while abs(temp_y_div % 1024) < epsilon and temp_y_div >= 1024 and div_i < len(size_name) - 1:
        temp_y_div /= 1024.0
        div_i += 1
    
    if div_i > 0 and abs(temp_y_div - round(temp_y_div)) < epsilon:
        y_fmt = round(temp_y_div)
        i_fmt = div_i
    else:
        temp_y_general = y
        i_general = 0
        while temp_y_general >= 1024 and i_general < len(size_name) - 1:
            temp_y_general /= 1024.0
            i_general += 1
        y_fmt = temp_y_general
        i_fmt = i_general

    if abs(y_fmt - round(y_fmt)) < epsilon:
        return f"{round(y_fmt):,}{size_name[i_fmt]}"
    else:
        return f"{y_fmt:,.1f}{size_name[i_fmt]}"


# Default width for printed / plotted branch names (0 = never shorten).
# Set from --name_width in main().
NAME_WIDTH = 70


def shorten_name(name, width):
    """
    Shorten one name to at most `width` characters, keeping BOTH the start
    (collection / container) and the end (the field that tells siblings apart):
        EventData/InDetTrackParticlesAux:.defin...CovMatrixDiag._0
    width <= 0 or a name that already fits is returned unchanged.
    """
    name = str(name)
    if width <= 0 or len(name) <= width:
        return name
    if width <= 8:
        return name[:width]
    budget = width - 3                       # room for "..."
    head = int(round(budget * 0.45))
    tail = budget - head
    return name[:head] + "..." + name[len(name) - tail:]


def unique_short_names(names, width=None):
    """
    Shorten every name to about `width` characters such that no two DIFFERENT
    names print the same. Names that still collide get more characters (in
    steps) until they differ, up to their full length. Returns a list parallel
    to `names`.
    """
    width = NAME_WIDTH if width is None else width
    names = [str(n) for n in names]
    if width <= 0:
        return names
    cur = {n: width for n in set(names)}
    short = {n: shorten_name(n, w) for n, w in cur.items()}
    while True:
        groups = {}
        for n, s in short.items():
            groups.setdefault(s, []).append(n)
        clashing = [g for g in groups.values() if len(g) > 1]
        if not clashing:
            break
        grew = False
        for g in clashing:
            for n in g:
                if cur[n] < len(n):
                    cur[n] = min(len(n), cur[n] + 6)
                    short[n] = shorten_name(n, cur[n])
                    grew = True
        if not grew:
            break
