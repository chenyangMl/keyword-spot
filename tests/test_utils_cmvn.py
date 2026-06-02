import json
import math
import os
import tempfile

import numpy as np

from model_convert.utils.cmvn import load_cmvn, load_kaldi_cmvn


class TestLoadCmvn:
    def test_basic(self):
        stats = {
            "mean_stat": [10.0, 20.0, 30.0],
            "var_stat": [200.0, 800.0, 1800.0],
            "frame_num": 2
        }
        # means: [5.0, 10.0, 15.0]
        # variance[i] = var_stat[i]/count - means[i]^2
        # variance[0] = 200/2 - 25 = 75 => istd = 1/sqrt(75)
        # variance[1] = 800/2 - 100 = 300 => istd = 1/sqrt(300)
        # variance[2] = 1800/2 - 225 = 675 => istd = 1/sqrt(675)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(stats, f)
            path = f.name
        try:
            cmvn = load_cmvn(path)
            assert cmvn.shape == (2, 3)
            np.testing.assert_allclose(cmvn[0], [5.0, 10.0, 15.0])
            np.testing.assert_allclose(cmvn[1, 0], 1.0 / math.sqrt(75), rtol=1e-5)
            np.testing.assert_allclose(cmvn[1, 1], 1.0 / math.sqrt(300), rtol=1e-5)
            np.testing.assert_allclose(cmvn[1, 2], 1.0 / math.sqrt(675), rtol=1e-5)
        finally:
            os.unlink(path)

    def test_very_small_variance_clamped(self):
        stats = {
            "mean_stat": [0.0],
            "var_stat": [0.0],
            "frame_num": 1
        }
        # variance = 0/1 - 0 = 0 => clamped to 1e-20 => istd = 1/sqrt(1e-20)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(stats, f)
            path = f.name
        try:
            cmvn = load_cmvn(path)
            expected_istd = 1.0 / math.sqrt(1.0e-20)
            np.testing.assert_allclose(cmvn[1, 0], expected_istd, rtol=1e-5)
        finally:
            os.unlink(path)


class TestLoadKaldiCmvn:
    def test_basic(self):
        # Format: <Tag> output_dim input_dim
        # Splice: splice_list_len * segs[2] == segs[1] => 2 * 10 = 20
        # AddShift: len(means) == segs[1] => 20
        # Rescale: len(variance) == segs[1] => 20
        content = (
            "<Splice> 20 10\n"
            "[ 0 1 ]\n"
            "<AddShift> 20 20\n"
            "[ -1.0 -2.0 -3.0 -4.0 -5.0 -6.0 -7.0 -8.0 -9.0 -10.0"
            " -11.0 -12.0 -13.0 -14.0 -15.0 -16.0 -17.0 -18.0 -19.0 -20.0 ]\n"
            "<Rescale> 20 20\n"
            "[ 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
            " 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 ]\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            path = f.name
        try:
            cmvn = load_kaldi_cmvn(path)
            # Splice has 2 elements => copy_times=2
            # means from AddShift are negated: [1..20]
            # The result should be tiled 2 times => shape (2, 40)
            assert cmvn.shape == (2, 40)
            # First 20 of row 0 = negated AddShift values = [1..20]
            np.testing.assert_allclose(cmvn[0, :20],
                                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
            # First 20 of row 1 = Rescale values
            np.testing.assert_allclose(cmvn[1, :20],
                                       [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                        1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
        finally:
            os.unlink(path)
