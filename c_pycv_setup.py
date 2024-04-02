from setuptools import setup, Extension
import numpy

setup(
    name="C_PyCv",
    version="0.5",
    packages=['pycv._lib._src'],
    ext_modules=[
        Extension(
            "pycv._lib._src.c_pycv",
            [
                "pycv/_lib/_src/c_pycv.c",
                "pycv/_lib/_src/c_pycv_base.c",
                "pycv/_lib/_src/c_pycv_filters.c",
                "pycv/_lib/_src/c_pycv_morphology.c",
                "pycv/_lib/_src/c_pycv_transform.c",
                "pycv/_lib/_src/c_pycv_canny.c",
                "pycv/_lib/_src/c_pycv_draw.c",
                "pycv/_lib/_src/c_pycv_measure.c",
                "pycv/_lib/_src/c_pycv_kd_tree.c",
                "pycv/_lib/_src/c_pycv_convex_hull.c",
                "pycv/_lib/_src/c_pycv_cluster.c",
                "pycv/_lib/_src/c_pycv_minmax_tree.c",
                "pycv/_lib/_src/c_pycv_haar_like.c",
            ],
            include_dirs=[numpy.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
        ),
    ]
)