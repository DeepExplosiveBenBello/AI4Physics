import numpy as np
from src.current_jumps import std_1D
from src.current_jumps import get_n_moms_of_moving_array
from utilities.test_utilities import list_to_test_std_1D
from utilities.test_utilities import list_to_test_get_n_moms_of_moving_array
import pytest

@pytest.mark.parametrize("inputs, expected_result", list_to_test_std_1D())
def test_std_1D(inputs, expected_result):
    result = std_1D(*inputs)
    assert result == expected_result

@pytest.mark.parametrize("inputs, expected_result",
                         list_to_test_get_n_moms_of_moving_array())
def test_get_n_moms_of_moving_array(inputs, expected_result):
    result = get_n_moms_of_moving_array(*inputs)
    assert np.array_equal(result, expected_result)
