import numpy as np


def list_to_test_std_1D():
    """
    Structures some test inputs of the function std_1D() under test along with
    the correct expected results. The return can be fed as argument directly
    into the decorator pytest.mark.parametrize() on the test function
    test_std_1D().

    Returns
    -------
    list_of_tests : list
        DESCRIPTION.
    list of tuples: [(inputs_0, exp_result_0),(inputs_1, exp_result_1), ...]

    """
    # inputs = array, mean(array)
    # expected_result = std(array)
    inputs_0 = np.array([1.0, 2.0, 2.0, 1]), 1.5
    exp_result_0 = 0.5
    inputs_1 = np.ones(4), 1.0
    exp_result_1 = 0.0
    # list_of_tests=[(inputs_0, exp_result_0),(inputs_1, exp_result_1), ...]
    list_of_tests = [(inputs_0, exp_result_0), (inputs_1, exp_result_1)]
    return list_of_tests


def list_to_test_get_n_moms_of_moving_array():
    """
    Structures some test inputs of the function get_n_moms_of_moving_array()
    under test along with the correct expected results.
    The return can be fed as argument directly into the decorator
    pytest.mark.parametrize() on the test function get_n_moms_of_moving_array()

    Returns
    -------
    list_of_tests : list
        DESCRIPTION.
    list of tuples: [(inputs_0, exp_result_0),(inputs_1, exp_result_1), ...]

    """
    # inputs = array, lag, window, n_moms
    # expected_result = array of MxN size
    inputs_0 = np.array([1.0, 1.0, 1.0, 2.0]), 1, 2, 2
    exp_result_0 = np.array(([1., 0.], [1., 0.], [1.5, 0.5]))
    inputs_1 = np.ones(6), 3, 3, 3
    exp_result_1 = np.array(([1., 0., 0.], [1., 0., 0.]))
    # list_of_tests=[(inputs_0, exp_result_0),(inputs_1, exp_result_1), ...]
    list_of_tests = [(inputs_0, exp_result_0), (inputs_1, exp_result_1)]
    return list_of_tests


def list_to_test_spectral_density():
    """
    Structures some test inputs of the function spectral_density() under test
    along with the correct expected results.
    The return can be fed as argument directly into the decorator
    pytest.mark.parametrize() on the test function test_spectral_density().

    Returns
    -------
    list_of_tests : list
        DESCRIPTION.
    list of tuples: [(inputs_0, exp_result_0),(inputs_1, exp_result_1), ...]

    """
    # inputs = array, lag, window
    # expected_result = std(array)
    inputs_0 = np.array([1.0, 1.0, 1.0, 2.0]), 1, 2
    exp_result_0 = np.array([[2.], [2.], [4.5]])
    inputs_1 = np.zeros(12), 3, 4
    exp_result_1 = np.zeros((4, 2))
    # list_of_tests=[(inputs_0, exp_result_0),(inputs_1, exp_result_1), ...]
    list_of_tests = [(inputs_0, exp_result_0), (inputs_1, exp_result_1)]
    return list_of_tests


def list_to_test_norm_auto_corr_pos_lags():
    """
    Structures some test inputs of the function norm_auto_corr_pos_lags()
    under test along with the correct expected results. The return can be
    fed as argument directly into the decorator pytest.mark.parametrize()
    on the test function test_norm_auto_corr_pos_lags().

    Returns
    -------
    list_of_tests : list
        DESCRIPTION.
    list of tuples: [(inputs_0, exp_result_0),(inputs_1, exp_result_1), ...]

    """
    # inputs = array, mean(array)
    # expected_result = std(array)
    inputs_0 = 2*np.ones(4), 0, 4
    exp_result_0 = np.array([1., 0.75])
    inputs_1 = 2*np.ones(12), 4, 4
    exp_result_1 = np.array([[1., 0.75], [1., 0.75], [1., 0.75]])
    # list_of_tests=[(inputs_0, exp_result_0),(inputs_1, exp_result_1), ...]
    list_of_tests = [(inputs_0, exp_result_0), (inputs_1, exp_result_1)]
    return list_of_tests
