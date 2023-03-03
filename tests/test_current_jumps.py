import numpy as np
from src.current_jumps import std_1D
from src.current_jumps import get_n_moms_of_moving_array
from tests.utilities.test_utilities import list_to_test_std_1D
from tests.utilities.test_utilities import list_to_test_get_n_moms_of_moving_array
from tests.utilities.test_utilities import list_to_test_spectral_density
from tests.utilities.test_utilities import list_to_norm_auto_corr_pos_lags
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

@pytest.mark.parametrize("inputs, expected_result",
                         list_to_test_spectral_density())
def test_spectral_density(inputs, expected_result):
    result = get_n_moms_of_moving_array(*inputs)
    assert np.array_equal(result, expected_result)
    
@pytest.mark.parametrize("inputs, expected_result",
                         list_to_norm_auto_corr_pos_lags())
def norm_auto_corr_pos_lags(inputs, expected_result):
    result = get_n_moms_of_moving_array(*inputs)
    assert np.array_equal(result, expected_result)