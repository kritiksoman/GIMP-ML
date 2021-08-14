import numpy as np
import cv2


def smooth_with_function_and_mask(image, function, mask, sigma):
    bleed_over = function(mask.astype(float), sigma)
    masked_image = np.zeros(image.shape, image.dtype)
    masked_image[mask] = image[mask]
    smoothed_image = function(masked_image, sigma)
    output_image = smoothed_image / (bleed_over + np.finfo(float).eps)
    return output_image


def generate_binary_structure(rank, connectivity):
    if connectivity < 1:
        connectivity = 1
    if rank < 1:
        return np.array(True, dtype=bool)
    output = np.fabs(np.indices([3] * rank) - 1)
    output = np.add.reduce(output, 0)
    return output <= connectivity


def canny(
    image,
    sigma=1.0,
    low_threshold=None,
    high_threshold=None,
    mask=None,
    use_quantiles=False,
):
    if low_threshold is None:
        low_threshold = 0.1
    elif use_quantiles:
        if not (0.0 <= low_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")

    if high_threshold is None:
        high_threshold = 0.2
    elif use_quantiles:
        if not (0.0 <= high_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")

    if mask is None:
        mask = np.ones(image.shape, dtype=bool)

    def fsmooth(x, sigma):
        return cv2.GaussianBlur(x, (0, 0), sigma)
        # return img_as_float(gaussian(x, sigma, mode='constant'))

    smoothed = smooth_with_function_and_mask(image, fsmooth, mask, sigma)
    jsobel = cv2.Sobel(np.float32(smoothed), cv2.CV_64F, 1, 0, 3)
    isobel = cv2.Sobel(np.float32(smoothed), cv2.CV_64F, 0, 1, 3)
    # jsobel = ndi.sobel(smoothed, axis=1)
    # isobel = ndi.sobel(smoothed, axis=0)
    abs_isobel = np.abs(isobel)
    abs_jsobel = np.abs(jsobel)
    magnitude = np.hypot(isobel, jsobel)

    #
    # Make the eroded mask. Setting the border value to zero will wipe
    # out the image edges for us.
    #
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.erode(image, kernel)
    s = generate_binary_structure(2, 2)
    s = np.array(s, dtype=np.uint8)
    eroded_mask = cv2.erode(mask.astype(np.uint8), s)
    # eroded_mask = binary_erosion(mask, s, border_value=0)
    eroded_mask = eroded_mask & (magnitude > 0)
    #
    # --------- Find local maxima --------------
    #
    # Assign each point to have a normal of 0-45 degrees, 45-90 degrees,
    # 90-135 degrees and 135-180 degrees.
    #
    local_maxima = np.zeros(image.shape, bool)
    # ----- 0 to 45 degrees ------
    pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
    pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    # Get the magnitudes shifted left to make a matrix of the points to the
    # right of pts. Similarly, shift left and down to get the points to the
    # top right of pts.
    c1 = magnitude[:, :][pts[:, :]]
    c2 = magnitude[:, :][pts[:, :]]
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:, :][pts[:, :]]
    c2 = magnitude[:, :][pts[:, :]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    # ----- 45 to 90 degrees ------
    # Mix diagonal and vertical
    #
    pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
    pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:, :][pts[:, :]]
    c2 = magnitude[:, :][pts[:, :]]
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:, :][pts[:, :]]
    c2 = magnitude[:, :][pts[:, :]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    # ----- 90 to 135 degrees ------
    # Mix anti-diagonal and vertical
    #
    pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
    pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1a = magnitude[:, :][pts[:, :]]
    c2a = magnitude[:, :][pts[:, :]]
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts]
    c_plus = c2a * w + c1a * (1.0 - w) <= m
    c1 = magnitude[:, :][pts[:, :]]
    c2 = magnitude[:, :][pts[:, :]]
    c_minus = c2 * w + c1 * (1.0 - w) <= m
    local_maxima[pts] = c_plus & c_minus
    # ----- 135 to 180 degrees ------
    # Mix anti-diagonal and anti-horizontal
    #
    pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
    pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:, :][pts[:, :]]
    c2 = magnitude[:, :][pts[:, :]]
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[:, :][pts[:, :]]
    c2 = magnitude[:, :][pts[:, :]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    local_maxima[pts] = c_plus & c_minus

    #
    # ---- If use_quantiles is set then calculate the thresholds to use
    #
    if use_quantiles:
        high_threshold = np.percentile(magnitude, 100.0 * high_threshold)
        low_threshold = np.percentile(magnitude, 100.0 * low_threshold)

    #
    # ---- Create two masks at the two thresholds.
    #
    high_mask = local_maxima & (magnitude >= high_threshold)
    low_mask = local_maxima & (magnitude >= low_threshold)

    #
    # Segment the low-mask, then only keep low-segments that have
    # some high_mask component in them
    #
    count, labels = cv2.connectedComponents(low_mask.astype(np.uint8))
    # strel = np.ones((3, 3), bool)
    # labels, count = label(low_mask, strel)
    if count == 0:
        return low_mask

    sums = np.array(
        np.sum(high_mask, labels, np.arange(count, dtype=np.int32) + 1),
        copy=False,
        ndmin=1,
    )
    good_label = np.zeros((count + 1,), bool)
    good_label[1:] = sums > 0
    output_mask = good_label[labels]
    return output_mask
