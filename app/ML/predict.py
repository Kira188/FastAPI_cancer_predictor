from keras_cv_attention_models import convnext
import tensorflow as tf
import pydicom
import cv2
import os
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent.parent
BASE_DIR = str(BASE_DIR)
print("Base Dir is:"+BASE_DIR)
# Configs
TARGET_HEIGHT = 1344
TARGET_WIDTH = 768
N_CHANNELS = 1
INPUT_SHAPE = (TARGET_HEIGHT, TARGET_WIDTH, N_CHANNELS)
TARGET_HEIGHT_WIDTH_RATIO = TARGET_HEIGHT / TARGET_WIDTH
THRESHOLD_BEST = 0.50
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
CROP_IMAGE = True
APPLY_CLAHE = False
APPLY_EQ_HIST = False
IMAGE_FORMAT = 'jpg'

#normalizing image 
def normalize(image):
    # Repeat channels to create 3 channel images required by pretrained ConvNextV2 models
    image = tf.repeat(image, repeats=3, axis=3)
    # Cast to float 32
    image = tf.cast(image, tf.float32)
    # Normalize with respect to ImageNet mean/std
    image = tf.keras.applications.imagenet_utils.preprocess_input(image, mode='torch')

    return image
def get_model():
    # Inputs, note the names are equal to the dictionary keys in the dataset
    image = tf.keras.layers.Input(INPUT_SHAPE, name='image', dtype=tf.uint8)

    # Normalize Input
    image_norm = normalize(image)

    # CNN Feature Maps
    x = convnext.ConvNeXtV2Tiny(
        input_shape=(TARGET_HEIGHT, TARGET_WIDTH, 3),
        pretrained=None,
        num_classes=0,
    )(image_norm)

    # Average Pooling BxHxWxC -> BxC
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Dropout to prevent Overfitting
    x = tf.keras.layers.Dropout(0.30)(x)
    # Output value between [0, 1] using Sigmoid function
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Define model with inputs and outputs
    model = tf.keras.models.Model(inputs=image, outputs=outputs)

    # Load pretrained Model Weights
    model_weights = os.path.join(os.path.dirname(__file__), 'model_weights.h5')
    model.load_weights(model_weights)
    # Set model non-trainable
    model.trainable = False

    # Compile model
    model.compile()

    return model
# Pretrained File Path: '/kaggle/input/sartorius-training-dataset/model.h5'
tf.keras.backend.clear_session()
# enable XLA optmizations
tf.config.optimizer.set_jit(True)

model = get_model()

# Source: https://www.kaggle.com/code/bobdegraaf/dicomsdl-voi-lut
# Voi Lut for image focusing
def voi_lut(image, dicom):
    # Additional Checks
    if 'WindowWidth' not in dicom.getPixelDataInfo() or 'WindowWidth' not in dicom.getPixelDataInfo():
        return image

    # Load only the variables we need
    center = dicom['WindowCenter']
    width = dicom['WindowWidth']
    bits_stored = dicom['BitsStored']
    voi_lut_function = dicom['VOILUTFunction']

    # For sigmoid it's a list, otherwise a single value
    if isinstance(center, list):
        center = center[0]
    if isinstance(width, list):
        width = width[0]

    # Set y_min, max & range
    y_min = 0
    y_max = float(2**bits_stored - 1)
    y_range = y_max

    # Function with default LINEAR (so for Nan, it will use linear)
    if voi_lut_function == 'SIGMOID':
        image = y_range / (1 + np.exp(-4 * (image - center) / width)) + y_min
    else:
        # Checks width for < 1 (in our case not necessary, always >= 750)
        center -= 0.5
        width -= 1

        below = image <= (center - width / 2)
        above = image > (center + width / 2)
        between = np.logical_and(~below, ~above)

        image[below] = y_min
        image[above] = y_max
        if between.any():
            image[between] = (
                ((image[between] - center) / width + 0.5) * y_range + y_min
            )

    return image

# Smooth vector used to smoothen sums/stds of axes
def smooth(l):
    # kernel size is 1% of vector
    kernel_size = int(len(l) * 0.01)
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(l, kernel, mode='same')

# X Crop offset based on first column with sum below 5% of maximum column sums*std
def get_x_offset(image, max_col_sum_ratio_threshold=0.05, debug=None):
    # Image Dimensions
    H, W = image.shape
    # Percentual margin added to offset
    margin = int(image.shape[1] * 0.00)
    # Threshold values based on smoothed sum x std to capture varying intensity columns
    vv = smooth(image.sum(axis=0).squeeze()) * smooth(image.std(axis=0).squeeze())
    # Find maximum sum in first 75% of columns
    vv_argmax = vv[:int(image.shape[1] * 0.75)].argmax()
    # Threshold value
    vv_threshold = vv.max() * max_col_sum_ratio_threshold

    # Find first column after maximum column below threshold value
    for offset, v in enumerate(vv):
        # Start searching from vv_argmax
        if offset < vv_argmax:
            continue

        # Column below threshold value found
        if v < vv_threshold:
            offset = min(W, offset + margin)
            break

    if isinstance(debug, np.ndarray):
        #debug[1].imshow(image)
        debug[1].set_title('X Offset')
        vv_scale = H / vv.max() * 0.90
        # Values
        debug[1].plot(H - vv * vv_scale , c='red', label='vv')
        # Threshold
        debug[1].hlines(H - vv_threshold * vv_scale, 0, W -1, colors='orange', label='threshold')
        # Max Value
        debug[1].scatter(vv_argmax, H - vv[vv_argmax] * vv_scale, c='blue', s=100, label='Max', zorder=np.PINF)
        # First Column Below Threshold
        debug[1].scatter(offset, H - vv[offset] * vv_scale, c='purple', s=100, label='Offset', zorder=np.PINF)
        debug[1].set_ylim(H, 0)
        debug[1].legend()
        debug[1].axis('off')

    return offset

# Y Crop offset based on first bottom and top rows with sum below 10% of maximum row sum*std
def get_y_offsets(image, max_row_sum_ratio_threshold=0.10, debug=None):
    # Image Dimensions
    H, W = image.shape
    # Margin to add to offsets
    margin = 0
    # Threshold values based on smoothed sum x std to capture varying intensity columns
    vv = smooth(image.sum(axis=1).squeeze()) * smooth(image.std(axis=1).squeeze())
    # Find maximum sum * std row in inter quartile rows
    vv_argmax = int(image.shape[0] * 0.25) + vv[int(image.shape[0] * 0.25):int(image.shape[0] * 0.75)].argmax()
    # Threshold value
    vv_threshold = vv.max() * max_row_sum_ratio_threshold
    # Default crop offsets
    offset_bottom = 0
    offset_top = H

    # Bottom offset, search from argmax to bottom
    for offset in reversed(range(0, vv_argmax)):
        v = vv[offset]
        if v < vv_threshold:
            offset_bottom = offset
            break

    # if isinstance(debug, np.ndarray):
        #debug[2].imshow(image)
        #debug[2].set_title('Y Bottom Offset')
        # vv_scale = W / vv.max() * 0.90
        # # Values
        # debug[2].plot(vv * vv_scale, np.arange(H), c='red', label='vv')
        # # Threshold
        # debug[2].vlines(vv_threshold * vv_scale, 0, H -1, colors='orange', label='threshold')
        # # Max Value
        # debug[2].scatter(vv[vv_argmax] * vv_scale, vv_argmax, c='blue', s=100, label='Max', zorder=np.PINF)
        # # First Column Below Threshold
        # debug[2].scatter(vv[offset_bottom] * vv_scale, offset_bottom, c='purple', s=100, label='Offset', zorder=np.PINF)
        # debug[2].set_ylim(H, 0)
        # debug[2].legend()
        # debug[2].axis('off')

    # Top offset, search from argmax to top
    for offset in range(vv_argmax, H):
        v = vv[offset]
        if v < vv_threshold:
            offset_top = offset
            break

    # if isinstance(debug, np.ndarray):
    #     #debug[3].imshow(image)
    #     debug[3].set_title('Y Top Offset')
    #     vv_scale = W / vv.max() * 0.90
    #     # Values
    #     debug[3].plot(vv * vv_scale, np.arange(H) , c='red', label='vv')
    #     # Threshold
    #     debug[3].vlines(vv_threshold * vv_scale, 0, H -1, colors='orange', label='threshold')
    #     # Max Value
    #     debug[3].scatter(vv[vv_argmax] * vv_scale, vv_argmax, c='blue', s=100, label='Max', zorder=np.PINF)
    #     # First Column Below Threshold
    #     debug[3].scatter(vv[offset_top] * vv_scale, offset_top, c='purple', s=100, label='Offset', zorder=np.PINF)
    #     debug[2].set_ylim(H, 0)
    #     debug[3].legend()
    #     debug[3].axis('off')

    return max(0, offset_bottom - margin), min(image.shape[0], offset_top + margin)

# Crop image and pad offsets to target image height/width ratio to preserve information
def crop(image, size=None, debug=False):
    # Image dimensions
    H, W = image.shape
    # Compute x/bottom/top offsets
    x_offset = get_x_offset(image, debug=debug)
    offset_bottom, offset_top = get_y_offsets(image[:,:x_offset], debug=debug)
    # Crop Height and Width
    h_crop = offset_top - offset_bottom
    w_crop = x_offset

    # Pad crop offsets to target aspect ratio
    if size is not None:
        # Height too large, pad x offset
        if (h_crop / w_crop) > TARGET_HEIGHT_WIDTH_RATIO:
            x_offset += int(h_crop / TARGET_HEIGHT_WIDTH_RATIO - w_crop)
        else:
            # Height too small, pad bottom/top offsets
            offset_bottom -= int(0.50 * (w_crop * TARGET_HEIGHT_WIDTH_RATIO - h_crop))
            offset_bottom_correction = max(0, -offset_bottom)
            offset_bottom += offset_bottom_correction

            offset_top += int(0.50 * (w_crop * TARGET_HEIGHT_WIDTH_RATIO - h_crop))
            offset_top += offset_bottom_correction

    # Crop Image
    image = image[offset_bottom:offset_top:,:x_offset]

    return image

def process(file_path, size=(TARGET_WIDTH, TARGET_HEIGHT), crop_image=CROP_IMAGE, apply_clahe=APPLY_CLAHE, apply_eq_hist=APPLY_EQ_HIST, debug=False, save=False):
    dicom = pydicom.dcmread(file_path)
    
    # Access pixel data
    image = dicom.pixel_array

    # Save original image for debug purposes
    if debug:
        fig, axes = plt.subplots(1, 5, figsize=(20,10))
        image0 = np.copy(image)
        axes[0].imshow(image0)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
    else:
        axes = False

    # voi_lut
    try:
        image = voi_lut(image, dicom)
    except:
        pass

    # Some images have 0 values as highest intensity and need to be inverted
    if dicom.PhotometricInterpretation == 'MONOCHROME1':
        image = np.max(image) - image

    # Normalize [0,1] range
    image = (image - image.min()) / (image.max() - image.min())

    # Convert to uint8 image in range [0, 255]
    image = (image * 255).astype(np.uint8)

    # Flip T0 Left/Right Orientation
    h0, w0 = image.shape
    if image[:,int(-w0 * 0.10):].sum() > image[:,:int(w0 * 0.10)].sum():
        image = np.flip(image, axis=1)

    # Crop Image
    if crop_image:
        image = crop(image, debug=axes)

    # Resize
    if size is not None:
        # Pad black pixels to make square image
        h, w = image.shape
        if (h / w) > TARGET_HEIGHT_WIDTH_RATIO:
            pad = int(h / TARGET_HEIGHT_WIDTH_RATIO - w)
            image = np.pad(image, [[0,0], [0, pad]])
            h, w = image.shape
        else:
            pad = int(0.50 * (w * TARGET_HEIGHT_WIDTH_RATIO - h))
            image = np.pad(image, [[pad, pad], [0,0]])
            h, w = image.shape
        # Resize
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    # Apply CLAHE contrast enhancement
    if apply_clahe:
        image = CLAHE.apply(image)

     # Apply Histogram Equalization
    if apply_eq_hist:
        image = cv2.equalizeHist(image)

    # Show Processed Image
    if debug:
        #axes[4].imshow(image)
        axes[4].set_title('Processed Image')
        axes[4].axis('off')
        plt.show()

    # Save Only
    
    return image

def make_prediction(file_name, rm_file):
    path = os.path.join(BASE_DIR,file_name)
    #path ='/Users/kiranseenivasan/Documents/Cancer Project/AI--CANCER-PROJECT-py/uploads/0a395ff2-d281-4927-ae5f-d52407298ccf.dcm'
    print("predict path is" + path)
    final_img = process(path, crop_image=True, size=(TARGET_WIDTH, TARGET_HEIGHT), debug=False, save=False)
    final_img = np.expand_dims(final_img, axis=-1)  # Add channel dimension
    final_img = np.expand_dims(final_img, axis=0)
    prediction = model.predict_on_batch(final_img).squeeze()
    prediction_value = float(prediction)
    cancer = int(np.int8(prediction > THRESHOLD_BEST))
    accuracy = round(prediction_value * 100, 2)
    if rm_file:
        os.remove(path)
    return prediction_value,cancer,accuracy
