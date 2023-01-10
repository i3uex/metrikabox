#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
import utils

@pytest.mark.parametrize('duration', [5, 10, 12.5, 15, 20])
@pytest.mark.parametrize('sample_rate', [8000, 16000, 22050, 44100, 48000])
@pytest.mark.parametrize('window', [0.2, 0.25, 0.33, 0.5, 1, 2, 5])
@pytest.mark.parametrize('step', [0.2, 0.25, 0.33, 0.5, 1, 2, 5])
def test_windowing(duration, sample_rate, window, step):
    test_array = np.arange(sample_rate*duration)
    a = utils.apply_window(test_array, window=window, step=step, sr=sample_rate).squeeze()
    assert int(sample_rate*window) == a.shape[1]
    assert int(duration/step)+1 == a.shape[0]

