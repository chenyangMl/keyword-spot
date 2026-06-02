import os
import tempfile

import torch
import torch.nn as nn
import yaml

from model_convert.utils.checkpoint import load_checkpoint, save_checkpoint


class TestSaveCheckpoint:
    def test_save_and_load(self):
        model = nn.Linear(4, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pt')
            save_checkpoint(model, path, infos={'epoch': 5})

            assert os.path.exists(path)
            info_path = path.replace('.pt', '.yaml')
            assert os.path.exists(info_path)

            with open(info_path) as f:
                infos = yaml.safe_load(f)
            assert infos['epoch'] == 5

    def test_save_no_infos(self):
        model = nn.Linear(4, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pt')
            save_checkpoint(model, path)

            info_path = path.replace('.pt', '.yaml')
            with open(info_path) as f:
                infos = yaml.safe_load(f)
            assert infos == {} or infos is None

    def test_save_data_parallel(self):
        model = nn.DataParallel(nn.Linear(4, 2))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pt')
            save_checkpoint(model, path)
            assert os.path.exists(path)


class TestLoadCheckpoint:
    def test_load_basic(self):
        model = nn.Linear(4, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pt')
            torch.save(model.state_dict(), path)

            model2 = nn.Linear(4, 2)
            configs = load_checkpoint(model2, path)
            assert configs == {}

            # Verify weights are loaded
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.equal(p1, p2)

    def test_load_with_yaml_info(self):
        model = nn.Linear(4, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pt')
            torch.save(model.state_dict(), path)

            info_path = path.replace('.pt', '.yaml')
            with open(info_path, 'w') as f:
                yaml.dump({'lr': 0.001}, f)

            model2 = nn.Linear(4, 2)
            configs = load_checkpoint(model2, path)
            assert configs['lr'] == 0.001
