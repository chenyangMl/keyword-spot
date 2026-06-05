# Copyright (c) 2021 Binbin Zhang
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


import logging
import os
import re

import yaml
import torch


def _explain_state_dict_mismatch(missing_keys, unexpected_keys) -> str:
    """Turn a strict load_state_dict mismatch into actionable guidance.

    The most common case (see issue #5/#15) is converting a custom CTC model
    trained on the modelscope `speech_charctc_kws_phone-xiaoyun` base: the
    checkpoint bakes the global cmvn into the model and uses an identity
    classifier, but the conversion config builds `global_cmvn=None` and a
    LinearClassifier, so the architectures do not match.
    """
    hints = []
    missing = set(missing_keys)
    unexpected = set(unexpected_keys)

    if any(k.startswith('global_cmvn.') for k in unexpected):
        hints.append(
            "- The checkpoint has a baked-in `global_cmvn` (keys "
            "`global_cmvn.mean/istd`) but your config builds no cmvn module. "
            "Add a `cmvn` section to your config so a GlobalCMVN is created, "
            "e.g.\n"
            "    cmvn:\n"
            "      load_from_checkpoint: true   # reuse cmvn stored in the .pt\n"
            "      norm_var: true\n"
            "  or point `cmvn_file` at the model's global_cmvn file.")
    if any(k.startswith('classifier.') for k in missing):
        hints.append(
            "- Your config builds a classifier with weights "
            "(e.g. LinearClassifier) but the checkpoint has none. For CTC/FSMN "
            "models the backbone already outputs `output_dim`, so set the "
            "classifier to identity in your config, e.g.\n"
            "    classifier:\n"
            "      type: identity\n"
            "      dropout: 0.1")

    msg = ["Checkpoint and model architecture do not match."]
    if missing_keys:
        msg.append("Missing keys (in model, not in checkpoint): %s" %
                    sorted(missing_keys))
    if unexpected_keys:
        msg.append("Unexpected keys (in checkpoint, not in model): %s" %
                   sorted(unexpected_keys))
    if hints:
        msg.append("Likely fixes:")
        msg.extend(hints)
    msg.append("See docs/model_convert.md (自定义CTC模型转换) for details.")
    return "\n".join(msg)


def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    if torch.cuda.is_available():
        logging.info('Checkpoint: loading from checkpoint %s for GPU' % path)
        checkpoint = torch.load(path)
    else:
        logging.info('Checkpoint: loading from checkpoint %s for CPU' % path)
        checkpoint = torch.load(path, map_location='cpu')
    # Inspect mismatches first so we can raise a clear, actionable error
    # instead of the cryptic default RuntimeError (see issue #5/#15).
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint,
                                                          strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            _explain_state_dict_mismatch(missing_keys, unexpected_keys))
    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs


def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    '''
    Args:
        infos (dict or None): any info you want to save.
    '''
    logging.info('Checkpoint: save to checkpoint %s' % path)
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)
    info_path = re.sub('.pt$', '.yaml', path)
    if infos is None:
        infos = {}
    with open(info_path, 'w') as fout:
        data = yaml.dump(infos)
        fout.write(data)
