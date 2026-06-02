# Copyright (c) 2024 Yang Chen (cyang8050@163.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, List

import json
import yaml
import torch
import torch.nn as nn


def load_config(config_path: str) -> dict:
    """Load a configuration file (JSON or YAML).

    Args:
        config_path: Path to the config file (.json or .yaml/.yml)

    Returns:
        Parsed configuration dictionary.
    """
    if config_path.endswith(".json"):
        with open(config_path) as f:
            return json.load(f)
    else:
        with open(config_path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)


def unpack_input(
    input,
    cache_shape: Tuple[int, ...] = (0, 0, 0, 0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpack a (tensor, cache) tuple or return a default empty cache.

    Many FSMN-family modules accept either a plain tensor or a
    (tensor, cache) tuple.  This helper standardizes that pattern.

    Args:
        input: Either a torch.Tensor or a tuple of (tensor, cache).
        cache_shape: Shape for the default empty cache tensor when input
            is not a tuple.

    Returns:
        (tensor, cache) tuple.
    """
    if isinstance(input, tuple):
        return input[0], input[1]
    return input, torch.zeros(*cache_shape, dtype=torch.float)


def forward_with_cache(
    blocks: nn.ModuleList,
    x: torch.Tensor,
    in_cache: torch.Tensor = torch.zeros(0, 0, 0, dtype=torch.float),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a sequence of blocks that each consume/produce a cache slice.

    This pattern is repeated in TCN, TCNStack, and parts of MDTC.
    Each block must expose a `padding` attribute and accept
    ``(x, cache) -> (x, new_cache)``.

    Args:
        blocks: Module list of blocks to iterate.
        x: Input tensor (B, D, T).
        in_cache: Concatenated input cache (B, D, total_padding).

    Returns:
        (output, new_cache) where new_cache is the concatenation of all
        per-block output caches.
    """
    out_caches: List[torch.Tensor] = []
    offset = 0
    for block in blocks:
        if in_cache.size(0) > 0:
            c_in = in_cache[:, :, offset:offset + block.padding]
        else:
            c_in = torch.zeros(0, 0, 0)
        x, c_out = block(x, c_in)
        out_caches.append(c_out)
        offset += block.padding
    new_cache = torch.cat(out_caches, dim=2)
    return x, new_cache


def read_weight_matrix(
    fread,
    num_rows: int,
    num_cols: int,
) -> torch.Tensor:
    """Read a weight matrix from a Kaldi-format text file.

    Reads ``num_rows`` lines, each containing ``num_cols`` float values,
    and returns them as a (num_rows, num_cols) tensor.

    Args:
        fread: File handle to read from.
        num_rows: Number of rows to read.
        num_cols: Expected number of columns per row.

    Returns:
        A float32 tensor of shape (num_rows, num_cols).
    """
    weights = torch.zeros((num_rows, num_cols), dtype=torch.float32)
    for i in range(num_rows):
        line = fread.readline()
        splits = line.strip().strip('[]').strip().split()
        assert len(splits) == num_cols
        cols = torch.tensor([float(item) for item in splits],
                            dtype=torch.float32)
        weights[i, :] = cols
    return weights
