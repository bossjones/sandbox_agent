"""
This type stub file was generated by pyright.
"""

__all__ = ['binned_statistic', 'binned_statistic_2d', 'binned_statistic_dd']
BinnedStatisticResult = ...
def binned_statistic(x, values, statistic=..., bins=..., range=...): # -> BinnedStatisticResult:
    """
    Compute a binned statistic for one or more sets of data.

    This is a generalization of a histogram function.  A histogram divides
    the space into bins, and returns the count of the number of points in
    each bin.  This function allows the computation of the sum, mean, median,
    or other statistic of the values (or set of values) within each bin.

    Parameters
    ----------
    x : (N,) array_like
        A sequence of values to be binned.
    values : (N,) array_like or list of (N,) array_like
        The data on which the statistic will be computed.  This must be
        the same shape as `x`, or a set of sequences - each the same shape as
        `x`.  If `values` is a set of sequences, the statistic will be computed
        on each independently.
    statistic : string or callable, optional
        The statistic to compute (default is 'mean').
        The following statistics are available:

          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'std' : compute the standard deviation within each bin. This
            is implicitly calculated with ddof=0.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by NaN.
          * 'count' : compute the count of points within each bin.  This is
            identical to an unweighted histogram.  `values` array is not
            referenced.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * 'min' : compute the minimum of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'max' : compute the maximum of values for point within each bin.
            Empty bins will be represented by NaN.
          * function : a user-defined function which takes a 1D array of
            values, and outputs a single numerical statistic. This function
            will be called on the values in each bin.  Empty bins will be
            represented by function([]), or NaN if this returns an error.

    bins : int or sequence of scalars, optional
        If `bins` is an int, it defines the number of equal-width bins in the
        given range (10 by default).  If `bins` is a sequence, it defines the
        bin edges, including the rightmost edge, allowing for non-uniform bin
        widths.  Values in `x` that are smaller than lowest bin edge are
        assigned to bin number 0, values beyond the highest bin are assigned to
        ``bins[-1]``.  If the bin edges are specified, the number of bins will
        be, (nx = len(bins)-1).
    range : (float, float) or [(float, float)], optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(x.min(), x.max())``.  Values outside the range are
        ignored.

    Returns
    -------
    statistic : array
        The values of the selected statistic in each bin.
    bin_edges : array of dtype float
        Return the bin edges ``(length(statistic)+1)``.
    binnumber: 1-D ndarray of ints
        Indices of the bins (corresponding to `bin_edges`) in which each value
        of `x` belongs.  Same length as `values`.  A binnumber of `i` means the
        corresponding value is between (bin_edges[i-1], bin_edges[i]).

    See Also
    --------
    numpy.digitize, numpy.histogram, binned_statistic_2d, binned_statistic_dd

    Notes
    -----
    All but the last (righthand-most) bin is half-open.  In other words, if
    `bins` is ``[1, 2, 3, 4]``, then the first bin is ``[1, 2)`` (including 1,
    but excluding 2) and the second ``[2, 3)``.  The last bin, however, is
    ``[3, 4]``, which *includes* 4.

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    First some basic examples:

    Create two evenly spaced bins in the range of the given sample, and sum the
    corresponding values in each of those bins:

    >>> values = [1.0, 1.0, 2.0, 1.5, 3.0]
    >>> stats.binned_statistic([1, 1, 2, 5, 7], values, 'sum', bins=2)
    BinnedStatisticResult(statistic=array([4. , 4.5]),
            bin_edges=array([1., 4., 7.]), binnumber=array([1, 1, 1, 2, 2]))

    Multiple arrays of values can also be passed.  The statistic is calculated
    on each set independently:

    >>> values = [[1.0, 1.0, 2.0, 1.5, 3.0], [2.0, 2.0, 4.0, 3.0, 6.0]]
    >>> stats.binned_statistic([1, 1, 2, 5, 7], values, 'sum', bins=2)
    BinnedStatisticResult(statistic=array([[4. , 4.5],
           [8. , 9. ]]), bin_edges=array([1., 4., 7.]),
           binnumber=array([1, 1, 1, 2, 2]))

    >>> stats.binned_statistic([1, 2, 1, 2, 4], np.arange(5), statistic='mean',
    ...                        bins=3)
    BinnedStatisticResult(statistic=array([1., 2., 4.]),
            bin_edges=array([1., 2., 3., 4.]),
            binnumber=array([1, 2, 1, 2, 3]))

    As a second example, we now generate some random data of sailing boat speed
    as a function of wind speed, and then determine how fast our boat is for
    certain wind speeds:

    >>> rng = np.random.default_rng()
    >>> windspeed = 8 * rng.random(500)
    >>> boatspeed = .3 * windspeed**.5 + .2 * rng.random(500)
    >>> bin_means, bin_edges, binnumber = stats.binned_statistic(windspeed,
    ...                 boatspeed, statistic='median', bins=[1,2,3,4,5,6,7])
    >>> plt.figure()
    >>> plt.plot(windspeed, boatspeed, 'b.', label='raw data')
    >>> plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,
    ...            label='binned statistic of data')
    >>> plt.legend()

    Now we can use ``binnumber`` to select all datapoints with a windspeed
    below 1:

    >>> low_boatspeed = boatspeed[binnumber == 0]

    As a final example, we will use ``bin_edges`` and ``binnumber`` to make a
    plot of a distribution that shows the mean and distribution around that
    mean per bin, on top of a regular histogram and the probability
    distribution function:

    >>> x = np.linspace(0, 5, num=500)
    >>> x_pdf = stats.maxwell.pdf(x)
    >>> samples = stats.maxwell.rvs(size=10000)

    >>> bin_means, bin_edges, binnumber = stats.binned_statistic(x, x_pdf,
    ...         statistic='mean', bins=25)
    >>> bin_width = (bin_edges[1] - bin_edges[0])
    >>> bin_centers = bin_edges[1:] - bin_width/2

    >>> plt.figure()
    >>> plt.hist(samples, bins=50, density=True, histtype='stepfilled',
    ...          alpha=0.2, label='histogram of data')
    >>> plt.plot(x, x_pdf, 'r-', label='analytical pdf')
    >>> plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=2,
    ...            label='binned statistic of data')
    >>> plt.plot((binnumber - 0.5) * bin_width, x_pdf, 'g.', alpha=0.5)
    >>> plt.legend(fontsize=10)
    >>> plt.show()

    """
    ...

BinnedStatistic2dResult = ...
def binned_statistic_2d(x, y, values, statistic=..., bins=..., range=..., expand_binnumbers=...): # -> BinnedStatistic2dResult:
    """
    Compute a bidimensional binned statistic for one or more sets of data.

    This is a generalization of a histogram2d function.  A histogram divides
    the space into bins, and returns the count of the number of points in
    each bin.  This function allows the computation of the sum, mean, median,
    or other statistic of the values (or set of values) within each bin.

    Parameters
    ----------
    x : (N,) array_like
        A sequence of values to be binned along the first dimension.
    y : (N,) array_like
        A sequence of values to be binned along the second dimension.
    values : (N,) array_like or list of (N,) array_like
        The data on which the statistic will be computed.  This must be
        the same shape as `x`, or a list of sequences - each with the same
        shape as `x`.  If `values` is such a list, the statistic will be
        computed on each independently.
    statistic : string or callable, optional
        The statistic to compute (default is 'mean').
        The following statistics are available:

          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'std' : compute the standard deviation within each bin. This
            is implicitly calculated with ddof=0.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by NaN.
          * 'count' : compute the count of points within each bin.  This is
            identical to an unweighted histogram.  `values` array is not
            referenced.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * 'min' : compute the minimum of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'max' : compute the maximum of values for point within each bin.
            Empty bins will be represented by NaN.
          * function : a user-defined function which takes a 1D array of
            values, and outputs a single numerical statistic. This function
            will be called on the values in each bin.  Empty bins will be
            represented by function([]), or NaN if this returns an error.

    bins : int or [int, int] or array_like or [array, array], optional
        The bin specification:

          * the number of bins for the two dimensions (nx = ny = bins),
          * the number of bins in each dimension (nx, ny = bins),
          * the bin edges for the two dimensions (x_edge = y_edge = bins),
          * the bin edges in each dimension (x_edge, y_edge = bins).

        If the bin edges are specified, the number of bins will be,
        (nx = len(x_edge)-1, ny = len(y_edge)-1).

    range : (2,2) array_like, optional
        The leftmost and rightmost edges of the bins along each dimension
        (if not specified explicitly in the `bins` parameters):
        [[xmin, xmax], [ymin, ymax]]. All values outside of this range will be
        considered outliers and not tallied in the histogram.
    expand_binnumbers : bool, optional
        'False' (default): the returned `binnumber` is a shape (N,) array of
        linearized bin indices.
        'True': the returned `binnumber` is 'unraveled' into a shape (2,N)
        ndarray, where each row gives the bin numbers in the corresponding
        dimension.
        See the `binnumber` returned value, and the `Examples` section.

        .. versionadded:: 0.17.0

    Returns
    -------
    statistic : (nx, ny) ndarray
        The values of the selected statistic in each two-dimensional bin.
    x_edge : (nx + 1) ndarray
        The bin edges along the first dimension.
    y_edge : (ny + 1) ndarray
        The bin edges along the second dimension.
    binnumber : (N,) array of ints or (2,N) ndarray of ints
        This assigns to each element of `sample` an integer that represents the
        bin in which this observation falls.  The representation depends on the
        `expand_binnumbers` argument.  See `Notes` for details.


    See Also
    --------
    numpy.digitize, numpy.histogram2d, binned_statistic, binned_statistic_dd

    Notes
    -----
    Binedges:
    All but the last (righthand-most) bin is half-open.  In other words, if
    `bins` is ``[1, 2, 3, 4]``, then the first bin is ``[1, 2)`` (including 1,
    but excluding 2) and the second ``[2, 3)``.  The last bin, however, is
    ``[3, 4]``, which *includes* 4.

    `binnumber`:
    This returned argument assigns to each element of `sample` an integer that
    represents the bin in which it belongs.  The representation depends on the
    `expand_binnumbers` argument. If 'False' (default): The returned
    `binnumber` is a shape (N,) array of linearized indices mapping each
    element of `sample` to its corresponding bin (using row-major ordering).
    Note that the returned linearized bin indices are used for an array with
    extra bins on the outer binedges to capture values outside of the defined
    bin bounds.
    If 'True': The returned `binnumber` is a shape (2,N) ndarray where
    each row indicates bin placements for each dimension respectively.  In each
    dimension, a binnumber of `i` means the corresponding value is between
    (D_edge[i-1], D_edge[i]), where 'D' is either 'x' or 'y'.

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> from scipy import stats

    Calculate the counts with explicit bin-edges:

    >>> x = [0.1, 0.1, 0.1, 0.6]
    >>> y = [2.1, 2.6, 2.1, 2.1]
    >>> binx = [0.0, 0.5, 1.0]
    >>> biny = [2.0, 2.5, 3.0]
    >>> ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx, biny])
    >>> ret.statistic
    array([[2., 1.],
           [1., 0.]])

    The bin in which each sample is placed is given by the `binnumber`
    returned parameter.  By default, these are the linearized bin indices:

    >>> ret.binnumber
    array([5, 6, 5, 9])

    The bin indices can also be expanded into separate entries for each
    dimension using the `expand_binnumbers` parameter:

    >>> ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[binx, biny],
    ...                                 expand_binnumbers=True)
    >>> ret.binnumber
    array([[1, 1, 1, 2],
           [1, 2, 1, 1]])

    Which shows that the first three elements belong in the xbin 1, and the
    fourth into xbin 2; and so on for y.

    """
    ...

BinnedStatisticddResult = ...
def binned_statistic_dd(sample, values, statistic=..., bins=..., range=..., expand_binnumbers=..., binned_statistic_result=...):
    """
    Compute a multidimensional binned statistic for a set of data.

    This is a generalization of a histogramdd function.  A histogram divides
    the space into bins, and returns the count of the number of points in
    each bin.  This function allows the computation of the sum, mean, median,
    or other statistic of the values within each bin.

    Parameters
    ----------
    sample : array_like
        Data to histogram passed as a sequence of N arrays of length D, or
        as an (N,D) array.
    values : (N,) array_like or list of (N,) array_like
        The data on which the statistic will be computed.  This must be
        the same shape as `sample`, or a list of sequences - each with the
        same shape as `sample`.  If `values` is such a list, the statistic
        will be computed on each independently.
    statistic : string or callable, optional
        The statistic to compute (default is 'mean').
        The following statistics are available:

          * 'mean' : compute the mean of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'median' : compute the median of values for points within each
            bin. Empty bins will be represented by NaN.
          * 'count' : compute the count of points within each bin.  This is
            identical to an unweighted histogram.  `values` array is not
            referenced.
          * 'sum' : compute the sum of values for points within each bin.
            This is identical to a weighted histogram.
          * 'std' : compute the standard deviation within each bin. This
            is implicitly calculated with ddof=0. If the number of values
            within a given bin is 0 or 1, the computed standard deviation value
            will be 0 for the bin.
          * 'min' : compute the minimum of values for points within each bin.
            Empty bins will be represented by NaN.
          * 'max' : compute the maximum of values for point within each bin.
            Empty bins will be represented by NaN.
          * function : a user-defined function which takes a 1D array of
            values, and outputs a single numerical statistic. This function
            will be called on the values in each bin.  Empty bins will be
            represented by function([]), or NaN if this returns an error.

    bins : sequence or positive int, optional
        The bin specification must be in one of the following forms:

          * A sequence of arrays describing the bin edges along each dimension.
          * The number of bins for each dimension (nx, ny, ... = bins).
          * The number of bins for all dimensions (nx = ny = ... = bins).
    range : sequence, optional
        A sequence of lower and upper bin edges to be used if the edges are
        not given explicitly in `bins`. Defaults to the minimum and maximum
        values along each dimension.
    expand_binnumbers : bool, optional
        'False' (default): the returned `binnumber` is a shape (N,) array of
        linearized bin indices.
        'True': the returned `binnumber` is 'unraveled' into a shape (D,N)
        ndarray, where each row gives the bin numbers in the corresponding
        dimension.
        See the `binnumber` returned value, and the `Examples` section of
        `binned_statistic_2d`.
    binned_statistic_result : binnedStatisticddResult
        Result of a previous call to the function in order to reuse bin edges
        and bin numbers with new values and/or a different statistic.
        To reuse bin numbers, `expand_binnumbers` must have been set to False
        (the default)

        .. versionadded:: 0.17.0

    Returns
    -------
    statistic : ndarray, shape(nx1, nx2, nx3,...)
        The values of the selected statistic in each two-dimensional bin.
    bin_edges : list of ndarrays
        A list of D arrays describing the (nxi + 1) bin edges for each
        dimension.
    binnumber : (N,) array of ints or (D,N) ndarray of ints
        This assigns to each element of `sample` an integer that represents the
        bin in which this observation falls.  The representation depends on the
        `expand_binnumbers` argument.  See `Notes` for details.


    See Also
    --------
    numpy.digitize, numpy.histogramdd, binned_statistic, binned_statistic_2d

    Notes
    -----
    Binedges:
    All but the last (righthand-most) bin is half-open in each dimension.  In
    other words, if `bins` is ``[1, 2, 3, 4]``, then the first bin is
    ``[1, 2)`` (including 1, but excluding 2) and the second ``[2, 3)``.  The
    last bin, however, is ``[3, 4]``, which *includes* 4.

    `binnumber`:
    This returned argument assigns to each element of `sample` an integer that
    represents the bin in which it belongs.  The representation depends on the
    `expand_binnumbers` argument. If 'False' (default): The returned
    `binnumber` is a shape (N,) array of linearized indices mapping each
    element of `sample` to its corresponding bin (using row-major ordering).
    If 'True': The returned `binnumber` is a shape (D,N) ndarray where
    each row indicates bin placements for each dimension respectively.  In each
    dimension, a binnumber of `i` means the corresponding value is between
    (bin_edges[D][i-1], bin_edges[D][i]), for each dimension 'D'.

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D

    Take an array of 600 (x, y) coordinates as an example.
    `binned_statistic_dd` can handle arrays of higher dimension `D`. But a plot
    of dimension `D+1` is required.

    >>> mu = np.array([0., 1.])
    >>> sigma = np.array([[1., -0.5],[-0.5, 1.5]])
    >>> multinormal = stats.multivariate_normal(mu, sigma)
    >>> data = multinormal.rvs(size=600, random_state=235412)
    >>> data.shape
    (600, 2)

    Create bins and count how many arrays fall in each bin:

    >>> N = 60
    >>> x = np.linspace(-3, 3, N)
    >>> y = np.linspace(-3, 4, N)
    >>> ret = stats.binned_statistic_dd(data, np.arange(600), bins=[x, y],
    ...                                 statistic='count')
    >>> bincounts = ret.statistic

    Set the volume and the location of bars:

    >>> dx = x[1] - x[0]
    >>> dy = y[1] - y[0]
    >>> x, y = np.meshgrid(x[:-1]+dx/2, y[:-1]+dy/2)
    >>> z = 0

    >>> bincounts = bincounts.ravel()
    >>> x = x.ravel()
    >>> y = y.ravel()

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> with np.errstate(divide='ignore'):   # silence random axes3d warning
    ...     ax.bar3d(x, y, z, dx, dy, bincounts)

    Reuse bin numbers and bin edges with new values:

    >>> ret2 = stats.binned_statistic_dd(data, -np.arange(600),
    ...                                  binned_statistic_result=ret,
    ...                                  statistic='mean')
    """
    ...
