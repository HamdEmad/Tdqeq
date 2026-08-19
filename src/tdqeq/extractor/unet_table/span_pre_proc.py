import cv2
import numpy as np


def calculate_contrast(img, img_mode) -> float:
    """
    Calculate the contrast of a given image.
    :param img: image, type is numpy.ndarray
    :param img_mode: image color channels, 'rgb' or 'bgr'
    :return: image contrast value
    """
    if img_mode == "rgb":
        # Convert RGB image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img_mode == "bgr":
        # Convert BGR image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Invalid image mode. Please provide 'rgb' or 'bgr'.")

    # Calculate mean and standard deviation
    mean_value = np.mean(gray_img)
    std_dev = np.std(gray_img)
    # Contrast is defined as standard deviation divided by mean (plus small constant to avoid division by zero)
    contrast = std_dev / (mean_value + 1e-6)
    return round(contrast, 2)
