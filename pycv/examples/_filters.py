import numpy as np
import os.path as osp
from pycv.io import ImageLoader, show_histograms
from pycv._lib.misc.histogram import histogram
from pycv.filters import hist_equalize, adjust_exp, adjust_log, adjust_gamma, adjust_linear


########################################################################################################################

def histogram_adjust():
    def plot_adjusts(_images, _titles=None):
        _his = [histogram(_img, bins=256) for _img in _images]
        show_histograms(
            _images,
            [_h.normalize[0] for _h in _his],
            [_h.bins for _h in _his],
            [_h.cdf()[0] for _h in _his],
            titles=_titles
        )

    loader = ImageLoader(osp.join(osp.split(osp.split(osp.dirname(__file__))[0])[0], '_debug_utils', 'data'))

    v_in = loader.load('moon')

    gamma_correction = adjust_gamma(v_in, gamma=2)
    log_correction = adjust_log(v_in)
    exp_correction = adjust_exp(v_in)

    plot_adjusts(
        [v_in, gamma_correction, log_correction, exp_correction],
        ['orig.', 'gamma correction', 'log correction', 'exp correction']
    )

    vl, vh = np.percentile(v_in, [2, 98])
    linear_correction = adjust_linear(v_in, in_range=(vl, vh), out_range=(0, 1))
    equalization = hist_equalize(v_in, n_bins=256)

    plot_adjusts(
        [v_in, linear_correction, equalization],
        ['orig.', 'linear correction', 'equalization']
    )
