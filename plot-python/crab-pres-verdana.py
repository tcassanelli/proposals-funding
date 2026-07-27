
import numpy as np

from scipy import fft, constants
import stingray.pulse

from astropy import units as u
from astropy.io import fits
from astropy.time import Time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from matplotlib import gridspec

plt.style.use('plot-python.mplstyle')
path_to_save = '.'
width = 5
height = width * .6
figsize = (width, height)

cmap = plt.cm.viridis

def binning(times: np.ndarray, dt : float, only_counts : bool=False):
    """
    dt: time-bin
    times_bins: bin edges
    """

    times_n = int(round((times[-1] / dt)) + 1)
    times_bins = np.linspace(0, times[-1], times_n, endpoint=False)
    timestream = np.histogram(times, bins=times_bins)[0]

    if only_counts:
        condition = (timestream != 0)
        times_bins = times_bins[:-1][condition]
        timestream = timestream[condition]

    return times_bins, timestream

def folding(times, dt, T, num_div):

    if T < dt:
        raise TypeError('Period (T) cannot be smaller than times bin (dt)')

    # Light-curve needs to have a division with no modulus
    M = num_div        # M for ease of notation
    N = round(T / dt)  # It will only select the integer value

    # Recalculating period with a (N + 1) step
    bins = np.linspace(0, T, N + 1)

    # number of samples that will be considered for each row of the waterfall
    ns = times.size // M

    # Modulus from division, it returns an element-wise remainder
    remainder = times % T

    waterfall = np.zeros((M, N), dtype=times.dtype)
    for m in range(M):
        indices = range(ns * m, ns * (m + 1))
        waterfall[m, :] = np.histogram(remainder[indices], bins=bins)[0]

    return remainder, waterfall


def periodogram(times : np.ndarray, dt : float, n_fft=None):
    """
    TODO: fix n_fft size. Note that padding with zeros is exactly the same as to use a larger FFT (in the complex case at least).
    """

    if n_fft is None:
        n_fft = fft.next_fast_len(times.size, real=False)

    N = int(n_fft / 2 + 1)
    spectral_density = np.abs(fft.fft(times, n=n_fft))[:N]
    frequencies = fft.fftfreq(n_fft, d=dt)[:N]

    return frequencies, spectral_density


def get_first_period(times, dt, sigma=5, low_floor=1):
    
    _, timestream = binning(times, dt)
    frequencies, spectral_density = periodogram(timestream, dt)
    
    noise_threshold = np.std(spectral_density) * sigma 
    for freq, power in zip(frequencies, spectral_density):
        if freq > low_floor and power > noise_threshold:
            return 1/freq
    else:
        raise ValueError("No significant period found in the data. Manually provide one setting self.period_init.")


def epoch_folding(
    times : float,
    period_init : float,
    dt : float,
    delta : float,
    n_iter : int,
    ) -> tuple:

    if period_init < dt:
        raise TypeError("Starting period cannot be smaller than time bin")

    nbin = int(np.round(period_init / dt))  # TODO: is this correct? 

    period_trials = np.linspace(
        start=period_init - delta * n_iter / 2,
        stop=period_init + delta * n_iter / 2,
        num=n_iter,
        endpoint=False,
        dtype=np.longdouble
        )
                
    frequencies = 1 / period_trials
    _, chi_squared_stats = stingray.pulse.epoch_folding_search(
        times=times,
        frequencies=frequencies,
        nbin=nbin
        )
    period_opt = period_trials[chi_squared_stats.argmax()]

    return period_opt, period_trials, chi_squared_stats


data = fits.open("/home/tcassanelli/iqueye/data/gs/bary/crab/QEYE_20250217-022750_crab1.baricentrizzato.tempo2/QEYE_20250217-022750_crab1.baricentrizzato.tempo2_uwct_0.fits")

times = Time(
    int(data[1].header["TMJDREF"].split(".")[0]) + data[1].data.astype(float),
    format="mjd",
    precision=9,
    scale="utc"
    )

start_time = times[0].copy()
print("start_time", start_time.isot)
dt_sp = 0.0008
# dt = (10 * u.us).to_value(u.s)

times = (times - start_time).to_value(u.s)

times_bins, timestream = binning(times, dt=dt_sp)
times_bins = times_bins[:-1]

idx_max = timestream.argmax()
times_bins -= times_bins[idx_max]

M = 200
dt_wf = (10 * u.us).to_value(u.s)
# T = 33.84140803 / 1000
# T = 0.0338409475465313
# T = 0.033841425484
T = 0.033838328546531300234

waterfall_raw = folding(times=times, dt=dt_wf, T=T, num_div=M)[1]
print("waterfall_raw.shape:", waterfall_raw.shape)

N = round(T / dt_wf)
nbins = np.linspace(0, T, N + 1)[:-1]
extent = [0, nbins.size, 0, M]

tlims = [-5, -5 + T * 1000] * u.ms

fig , ax = plt.subplots(figsize=figsize, dpi=200)

ax.set_xlabel(f"Time ms + {(start_time + times_bins[idx_max] * u.s).iso}")
ax.set_ylabel("Photon counts")

ax.set_xlim(tlims.to_value(u.s).tolist())
ax.set_ylim(timestream.min(), timestream.max() * 1.1)
ax.plot(times_bins, timestream, alpha=1)

fig.tight_layout()
fig.savefig("crab-pres-verdana.pdf")





