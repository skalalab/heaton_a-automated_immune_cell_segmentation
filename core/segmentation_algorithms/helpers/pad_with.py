def pad_with(vector, pad_width = 0, iaxis = 0, **kwargs):
    """
    Pad with: numpy pad helper function to pad a given array with a set value

    Taken directly from the numpy padding documentation: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        - Uniformly pads a given image with a set width of pixels

    Parameters
    ----------
    vector: list
        Image to be padded.
    pad_width: int, optional
        Width of pixels ot uniformly pad.
        Default is 0.
    iaxis: int, optional
        Axis currently be calculated.
        Default is 0.
    kwargs: dictionary, optional
        Arbitrary keyword arguments.

    Returns
    -------
    None
    """

    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value