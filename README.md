# PyCv - Image Processing Package

This is a Python package for image processing that includes various sub-packages and 
functions written in both Python and C extensions for enhanced performance. 
The package provides functionalities for manipulating images in different ways, 
including color conversion, drawing shapes, applying filters, measuring properties, 
performing morphological operations, segmentation, working with data structures, 
and image transformation.

## Sub-packages:

### 1. colors: Functions for color space conversions.

* **rgb2gray:** Converts RGB image to grayscale.
* **rgb2hsv:** Converts RGB image to HSV color space.
* **rgb2yuv:** Converts RGB image to YUV color space.
* **yuv2rgb:** Converts YUV image to RGB color space

### 2. draw: Functions for drawing shapes on images.

* **draw_circle:** Draws a circle on an image.
* **draw_ellipse:** Draws an ellipse on an image.
* **draw_line:** Draws a line on an image.

### 3. filters: Functions for applying different types of filters on images.

* **canny:** Applies Canny edge detection to an image.
* **edge:** Detects edges in an image using various algorithms.
* **gaussian_filter:** Applies Gaussian smoothing to an image.
* **generic:** Applies a custom filter to an image.
* **image_filter:** Applies a filter to an image.
* **local_max_filter:** Applies a local maximum filter to an image.
* **local_min_filter:** Applies a local minimum filter to an image.
* **mean_filter:** Applies a mean filter to an image.
* **median_filter:** Applies a median filter to an image.
* **prewitt:** Applies Prewitt edge detection to an image.
* **rank_filter:** Applies a rank filter to an image.
* **sobel:** Applies Sobel edge detection to an image

### 4. measurements: Functions for measuring properties of image regions.

* regionprops: Calculates properties of image regions.

### 5. morphological: Functions for morphological operations on images.

* **area_close:** Performs area closing operation on binary images.
* **area_open:** Performs area opening operation on binary images.
* **binary:** Converts a grayscale image to binary using a threshold.
* **binary_closing:** Performs binary closing operation on binary images.
* **binary_dilation:** Performs binary dilation operation on binary images.
* **binary_edge:** Detects edges in binary images.
* **binary_erosion:** Performs binary erosion operation on binary images.
* **binary_fill_holes:** Fills holes in binary images.
* **binary_hit_or_miss:** Performs hit-or-miss transformation on binary images.
* **binary_opening:** Performs binary opening operation on binary images.
* **black_top:** Applies black top-hat transform on grayscale images.
* **convex_hull:** Finds the convex hull of objects in binary images.
* **convex_image:** Generates an image from convex hull vertices.
* **find_object:** Finds objects in labeled images.
* **gray:** Converts binary images to grayscale.
* **gray_closing:** Performs grayscale closing operation on grayscale images.
* **gray_dilation:** Performs grayscale dilation operation on grayscale images.
* **gray_erosion:** Performs grayscale erosion operation on grayscale images.
* **gray_opening:** Performs grayscale opening operation on grayscale images.
* **im_label:** Labels connected components in binary images.
* **region_fill:** Fills regions in images starting from a seed point.
* **regions:** Finds connected regions in binary images.
* **remove_small_holes:** Removes small holes from binary images.
* **remove_small_objects:** Removes small objects from binary images.
* **skeletonize:** Reduces binary images to a skeleton representation.
* **white_top:** Applies white top-hat transform on grayscale images.

### 6. segmentation: Functions for image segmentation.

* **adaptive_threshold:** Applies adaptive thresholding to binarize images, where the threshold value varies across the image.
* **im_binarize:** Binarizes images using a specified threshold value.
* **im_threshold:** Binarizes images using a threshold value.
* **kapur_threshold:** Calculates the threshold value based on Kapur's entropy method for binarization.
* **li_and_lee_threshold:** Calculates the threshold value based on Li and Lee's method for binarization.
* **mean_threshold:** Calculates the threshold value based on mean intensity for binarization.
* **minimum_error_threshold:** Calculates the threshold value based on minimum error for binarization.
* **minimum_threshold:** Calculates the threshold value based on minimum intensity for binarization.
* **otsu_threshold:** Calculates the threshold value based on Otsu's method for binarization.
* **thresholding:** Performs thresholding on images using various thresholding techniques.

### 7. structures: Functions for handling spatial data structures.

* **kdtree:** Implementation of k-dimensional trees for spatial data structures.

### 8. transform: Functions for geometric transformations.

* **geometric_transform:** Applies geometric transformations to images, such as translation, rotation, scaling, and shearing, using custom transformation matrices.
* **hough_circle:** Detects circles in images using the Hough circle transform.
* **hough_circle_peak:** Identifies the peaks in the Hough space generated by the Hough circle transform, corresponding to the detected circles.
* **hough_line:** Detects straight lines in images using the Hough line transform.
* **hough_line_peak:** Identifies the peaks in the Hough space generated by the Hough line transform, corresponding to the detected lines.
* **hough_probabilistic_line:** Detects lines in images using the probabilistic Hough line transform, which is more efficient than the standard Hough line transform.
* **pyramids:** Functions for creating image pyramids, including Gaussian and Laplacian pyramids.
* **resize:** Resizes images to a specified shape using different interpolation methods, such as nearest neighbor, linear, quadratic, or cubic interpolation.
* **rotate:** Rotates images by a specified angle around a given axis using various interpolation methods.
* **AffineTransform:** Defines an affine transformation, which includes translation, rotation, scaling, and shearing, to be applied to images.
* **DOGPyramid:** Constructs a Difference of Gaussians (DoG) pyramid, which is used for scale-invariant feature detection.
* **GaussianPyramid:** Constructs a Gaussian pyramid, which is a multi-scale representation of an image used in computer vision and image processing tasks.
* **GaussianScaleSpace:** Constructs a Gaussian scale space, which is a series of smoothed and downsampled versions of an image used for blob detection and feature extraction.
* **LaplacianPyramid:** Constructs a Laplacian pyramid, which is derived from a Gaussian pyramid and is used for image compression, edge detection, and image blending.
* **ProjectiveTransform:** Defines a projective transformation, also known as a perspective transformation, to be applied to images.
* **RidgeTransform:** Applies ridge transformation to images, which enhances ridges and suppresses noise.
* **SimilarityTransform:** Defines a similarity transformation, which includes translation, rotation, and uniform scaling, to be applied to images.


## Requirements:
* Python 3.x
* NumPy
* C Compiler (for C extensions)

## Ongoing Development:
This image processing package is under active development. 
New functions and enhancements are continually being added to improve its capabilities and performance. 
If you have any suggestions for new features or improvements, please don't hesitate to reach out or
contribute directly.