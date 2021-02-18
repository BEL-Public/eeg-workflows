from typing import List
from datetime import datetime

from mffpy import Writer
from mffpy.writer import BinWriter

from .average import Average


def write_averaged(averages: List[Average], outfile: str,
                   startdatetime: datetime, device: str) -> Writer:
    """Write averaged EEG data to MFF

    Parameters
    ----------
    averages
        List of `Average` objects to be written
    outfile
        Path to which the averaged MFF will be written
    startdatetime
        Timestamp of recording start for the raw MFF from which the averages
        were generated
    device
        Recording device for the raw MFF from which the averages were generated

    Returns
    -------
    W
        The `mffpy.Writer` object

    Raises
    ------
    ValueError
        If averaged data blocks are not of equal shape
    ValueError
        If averages have differing sampling rates
    """
    for average in averages:
        if average.data().shape != averages[0].data().shape:
            raise ValueError('Averaged data blocks of different shape: '
                             f'{average.data().shape} != '
                             f'{averages[0].data().shape}')
        if average.sampling_rate != averages[0].sampling_rate:
            raise ValueError('Averages have different sampling rates: '
                             f'{average.sampling_rate} != '
                             f'{averages[0].sampling_rate}')

    W = Writer(outfile)
    W.addxml('fileInfo', recordTime=startdatetime)
    W.add_coordinates_and_sensor_layout(device=device)

    # Add EEG data
    sampling_rate = averages[0].sampling_rate
    eeg_bin = BinWriter(sampling_rate=int(sampling_rate))
    for average in averages:
        eeg_bin.add_block(average.data(), offset_us=0)
    W.addbin(eeg_bin)

    # Add category info
    categories_content = {}
    begin_time = 0
    for average in averages:
        categories_content[average.category] = [
            average.build_category_content(begin_time)
        ]
        begin_time += int(average.num_samples() / average.sampling_rate * 1e6)

    W.addxml('categories', categories=categories_content)

    W.write()
    return W
