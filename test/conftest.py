import numpy as np
import pytest


@pytest.fixture
def rgb_512():
    return np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def gray_512():
    return np.random.randint(0, 256, (512, 512), dtype=np.uint8)


@pytest.fixture
def small_rgb():
    return np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def rgb_1024():
    return np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)


@pytest.fixture
def all_zeros():
    return np.zeros((512, 512, 3), dtype=np.uint8)


@pytest.fixture
def all_max():
    return np.full((512, 512, 3), 255, dtype=np.uint8)
