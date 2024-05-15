from pycv.io import ImageLoader, DEFAULT_DATA_PATH, show_scale_space, show_pyramid
from pycv.transform import GaussianPyramid, GaussianScaleSpace


########################################################################################################################


loader = ImageLoader(DEFAULT_DATA_PATH)
inputs = loader.load('lena', _color_fmt='RGB2GRAY')


pyramid1 = GaussianPyramid(
    2,
    sigma=(1.5, 1.5),
    order=1,
    preserve_dtype=False
)

p1 = [i for i in pyramid1(inputs)]
show_pyramid(p1)


pyramid2 = GaussianScaleSpace(
    2,
    scalespace=[1.1, 1.5, 1.8, 2.1],
    rescale_index=-1,
    order=0,
    preprocess_scales=1.6,
    preprocess_factors=2.0,
    preprocess_order=1,
    preserve_dtype=False
)

p2 = [i for i in pyramid2(inputs)]
show_scale_space(p2)