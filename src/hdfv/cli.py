from glob import glob
from pathlib import Path
from typing import List, Optional, cast

import h5py
import numpy as np
import typer
from hdfx.base import parse_slice
from hdfx.merge import h5merge
from hdfx.shard import h5shard
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Unix-style tools for working with HDF5")
console = Console()
err_console = Console(stderr=True)
