import numpy as np
from src.current_jumps import std_1D
from src.current_jumps import get_n_moms_of_moving_array
import pytest

# list of inputs and expected results for std_1D() function
def list_to_test_std_1D():
    # inputs = array, mean(array)
    # expected_result = std(array)
    inputs_0 = np.array([1.0, 2.0, 2.0, 1]), 1.5
    exp_result_0 = 0.5
    inputs_1 = np.ones(4), 1.0
    exp_result_1 = 0.0
    # list_of_tests=[(inputs_0, exp_result_0),(inputs_1, exp_result_1), ...]
    list_of_tests = [(inputs_0, exp_result_0),(inputs_1, exp_result_1)]
    return list_of_tests

@pytest.mark.parametrize("inputs, expected_result", list_to_test_std_1D())
def test_std_1D(inputs, expected_result):
    result = std_1D(*inputs)
    assert result == expected_result

# list of inputs and expected results for get_n_moms_of_moving_array()
def list_to_test_get_n_moms_of_moving_array():
    # inputs = array, lag, window, n_moms
    # expected_result = array of MxN size
    inputs_0 = np.array([1.0, 1.0, 1.0, 2.0]), 1, 2, 2
    exp_result_0 = np.array(([1., 0.],[1., 0.],[1.5, 0.5]))
    inputs_1 = np.ones(6), 3, 3, 3
    exp_result_1 = np.array(([1., 0., 0.],[1., 0., 0.]))
    # list_of_tests=[(inputs_0, exp_result_0),(inputs_1, exp_result_1), ...]
    list_of_tests = [(inputs_0, exp_result_0),(inputs_1,exp_result_1)]
    return list_of_tests


@pytest.mark.parametrize("inputs, expected_result",
                         list_to_test_get_n_moms_of_moving_array())
def test_get_n_moms_of_moving_array(inputs, expected_result):
    result = get_n_moms_of_moving_array(*inputs)
    assert np.array_equal(result,expected_result)
