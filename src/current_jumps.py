import numpy as np
import pandas as pd

def standardization(array):
    """
    Standardize a 1D numpy array by subtracting its mean and dividing by its
    standard deviation.

    Parameters
    ----------
    array : 1D-numpy array

    Returns
    -------
    1D-numpy array

    """
    return (array - array.mean())/(array.std())

def differentiation(array):
    """
    Differenciate discretly a 1D numpy array by subtracting each element with
    the previous one. Notice: it duplicates the second last element with the
    last so that the array doesn't change dimention.

    Parameters
    ----------
    array : 1D-numpy array

    Returns
    -------
    1D-numpy array of equal dimension

    """
    array = np.diff(array)
    return np.append(array, array[-1])

def std_1D(array, mean):
    """
    Calculate the standard deviation of an array, by using its mean.
    
    Parameters
    ----------
    array : 1D-numpy array
    mean : the mean of the array

    Returns
    -------
    float

    """
    sum_of_ls = 0
    for value in array:
        sum_of_ls += (value - mean)**2
    return (sum_of_ls/len(array))**(0.5)

def central_n_mom_1D(array, mean, n):
    """
    Calculate the n-th moment of an array centralize to the mean by using its
    mean.
    
    Parameters
    ----------
    array : 1D-numpy array
    mean : float - the mean of the array
    n : integer - nth moment to be calculated

    Returns
    -------
    float

    """
    if n < 2 or not isinstance(n, int):
        raise ValueError("n must be an integer greater or equal 2")

    sum_of_n_ls = 0
    for value in array:
        sum_of_n_ls += (value-mean)**(n+1)
    return sum_of_n_ls/len(array)

def get_n_moms_of_moving_array(array, lag, window, n_moms):
    """
    This function creates a MxN-vector of moments, where:
    N(# of moments) = n_moms
    M(# of subarrays) = (len(array) - window)/lag + 1:
        
    INPUT ---------------------------------------------------------------
    array --> numerical numpy 1D-array of size n.
    window --> constant integer: # of points defining a subarray or series.
    lag --> each subarray is the previous one shifted by # of points = lag.
    n_moms --> nothing to do with mothers. # of momenta to calculates*. 
    
    *Notice: the first 2 moments are the mean and standard deviation.
    
    OUTPUT ---------------------------------------------------------------
    n_mom_vec --> numpy NxM-array
    """
    # sanity-check arguments
    if ((len(array) - window) % lag) != 0:
        raise ValueError("""The window and lag chosen must be such that the
                         difference between the size of the array and the
                         window is multiple of the lag.""")

    if not all(isinstance(i, int) for i in [lag, window, n_moms]):
        raise ValueError("""window, lag and n_moms arguments must be all
                         integers""")

    # number of series/subarray calculated by window and lag
    n_of_series = int((len(array) - window)/lag + 1)

    # initialization of arrays
    means_vec = np.zeros(n_of_series)
    std_vec = np.zeros(n_of_series)
    n_mom_vec = np.zeros((n_of_series, n_moms))

    # first series
    series_1st = array[0:int(window)]
    means_vec[0] = series_1st.mean()
    std_vec[0] = series_1st.std()
    n_mom_vec[0][0] = means_vec[0]
    n_mom_vec[0][1] = std_vec[0]

    # calculate and storing the 3th - (n-2)th moments
    if n_moms > 2:
        for n in range(2, n_moms):
            n_mom_vec[0][n] = central_n_mom_1D(series_1st, means_vec[0], n)

    # calculate all moments for rolling series
    for i in range(1, n_of_series):
        # the mean of a rolling series can be calculated faster by using
        # the mean of the previous series
        ith_series = array[lag*i:window+lag*i]
        mean_prev = means_vec[i-1]
        old_values = np.sum(array[lag*(i-1):lag*i])
        new_values = np.sum(array[window + lag*(i-1):window + lag*(i)])
        # calculating and storing mean, std and all moments or superior order
        means_vec[i] = mean_prev - old_values/window + new_values/window
        n_mom_vec[i][0] = means_vec[i]
        n_mom_vec[i][1] = std_1D(ith_series, means_vec[i])
        if n_moms > 2:
            for n in range(2, n_moms):
                n_mom_vec[i][n] = central_n_mom_1D(ith_series, means_vec[i], n)
    return n_mom_vec

# Execute code when the 
# file runs as a script, but not when it’s imported as a module
if __name__ == "__main__":
    # load data
    columns_names = ["Current (A)", "Timestamp (ms)", "id"]
    try:
        data_path = "../data/current_data.csv"
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        data_path = "data/current_data.csv"
        df = pd.read_csv(data_path)
    # some parameters
    window = 100
    lag = 20
    n_moments = 10

    # processing data
    data = pd.Series.to_numpy(df["Current (A)"])
    data = standardization(data)
    data_der = differentiation(data)

    # creating vector of features
    data_moments = get_n_moms_of_moving_array(data, lag, window, n_moments)
    data_der_moments = get_n_moms_of_moving_array(data_der, lag, window, n_moments)
