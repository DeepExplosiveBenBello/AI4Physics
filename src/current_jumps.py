import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler

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
    Create a MxN-vector of moments.

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


def norm_auto_corr_pos_lags(array, lag, window):
    """
    Build a MxN-vector of autocorrelation values.

    N(# of coefficients) = window//2
    M(# of subarrays) = (len(array) - window)/lag + 1:
    The vector will be normalized to the variance (first element) and only
    positive lags are considered.

    INPUT ---------------------------------------------------------------
    array --> numerical numpy 1D-array of size n.
    window --> constant integer: # of points defining a subarray or series.
    lag --> each subarray is the previous one shifted by # of points = lag.

    OUTPUT ---------------------------------------------------------------
    a_corr_vec --> numpy NxM-array
    """
    # number of series/subarray calculated by window and lag
    n_of_series = int((len(array) - window)/lag + 1)

    # initialization of arrays
    a_corr_vec = np.zeros((n_of_series, window//2))

    # calculate normalized autocorrelation with pos lags for all series
    for i in range(n_of_series):
        ith_series = array[lag*i:window+lag*i]
        # Autocorrelation of data series
        a_corr_full = signal.correlate(ith_series, ith_series, mode='same')
        a_corr_full /= len(ith_series)
        # Normalization lag in respect to the variance (1st element lag = 0)
        a_corr_full = a_corr_full/a_corr_full[0]
        # Only positive lags
        a_corr_vec[i] = a_corr_full[len(ith_series)//2:]

    return a_corr_vec


def spectral_density(array, lag, window):
    """
    Build a MxN-vector of power spectral values.

    N(# of power spectral values) --> N = window//2
    M(# of subarrays on which psd is performed) -->
    M = (len(array) - window)/lag + 1:
    Only positive frequencies are considered

    INPUT ---------------------------------------------------------------
    array --> numerical numpy 1D-array of size n > window.
    window --> constant integer: # of points defining a subarray or series.
    lag --> each subarray is the previous one shifted by # of points = lag.

    OUTPUT ---------------------------------------------------------------
    spec_dens_vec --> numpy MxN-array
    """
    # number of series/subarray calculated by window and lag
    n_of_series = int((len(array) - window)/lag + 1)

    # initialization of arrays
    spec_dens_vec = np.zeros((n_of_series, window//2))

    # calculate normalized autocorrelation with pos lags for all series
    for i in range(n_of_series):
        ith_series = array[lag*i:window+lag*i]

        # Compute the power spectral density using the numpy module
        sp_full = np.abs(np.fft.fft(ith_series))**2
        sp_full = sp_full/len(ith_series)
        spec_dens_vec[i] = sp_full[:len(sp_full)//2]
    return spec_dens_vec


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
lag = 100
n_moments = 10

# processing data
data = pd.Series.to_numpy(df["Current (A)"])
data = standardization(data)
data_der = differentiation(data)

# n moments of data series
data_moments = get_n_moms_of_moving_array(data, lag, window, n_moments)
# n moments of data differentiated series
data_der_moments = get_n_moms_of_moving_array(data_der, lag, window, n_moments)
# Autocorrelation of data series
data_a_cor = norm_auto_corr_pos_lags(data, lag, window)
# Autocorrelation of data differentiated series
data_der_a_cor = norm_auto_corr_pos_lags(data_der, lag, window)
# Power spectral density of data series
data_psd = spectral_density(data, lag, window)
# Power spectral density of data differentiated series
data_der_psd = spectral_density(data_der, lag, window)

# creating vector of features
features = np.concatenate((data_moments, data_der_moments, data_a_cor,
                           data_der_a_cor, data_psd, data_der_psd), axis=1)
# standardizing features
stand_features = StandardScaler().fit_transform(features)
