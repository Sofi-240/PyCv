import numpy as np
from pycv.io import ImageLoader, show_histograms, DEFAULT_DATA_PATH
from pycv._lib.misc.histogram import histogram
from pycv.filters import hist_equalize, adjust_exp, adjust_log, adjust_gamma, adjust_linear


########################################################################################################################

def _plot_adjusts_(_images, _titles=None):
    _his = [histogram(_img, bins=256) for _img in _images]
    show_histograms(
        _images,
        [_h.normalize[0] for _h in _his],
        [_h.bins for _h in _his],
        [_h.cdf()[0] for _h in _his],
        titles=_titles
    )


########################################################################################################################

loader = ImageLoader(DEFAULT_DATA_PATH)
v_in = loader.load('moon')


gamma_correction_bright = adjust_gamma(v_in, gamma=0.5)
gamma_correction_dark = adjust_gamma(v_in, gamma=2)

_plot_adjusts_(
    [v_in, gamma_correction_bright, gamma_correction_dark],
    ['orig.', 'gamma correction bright', 'gamma correction dark']
)


log_correction = adjust_log(v_in)
exp_correction = adjust_exp(v_in)

_plot_adjusts_(
    [v_in, log_correction, exp_correction],
    ['orig.', 'log correction', 'exp correction']
)

vl, vh = np.percentile(v_in, [5, 95])
linear_correction = adjust_linear(v_in, in_range=(vl, vh), out_range=(0, 1))

equalization = hist_equalize(v_in, n_bins=256)

_plot_adjusts_(
    [v_in, linear_correction, equalization],
    ['orig.', 'linear correction', 'equalization']
)


########################################################################################################################





