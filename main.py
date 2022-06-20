'''
Nazanin Moradinasab
This code converts multichannel immunoflourecsent images into RGB
'''

# import libraries
import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
import random
import colorsys

# path
tiff_path = '/Users/nazanin/Documents/Ph.D./Python_codes/Image_classification/TIFF_wsi/'
rgb_path = '/Users/nazanin/Documents/Ph.D./Python_codes/Image_classification/RGB_wsi_LGAL3_python/'


def main(img, img_name):
    n_channels = img.shape[2]
    # create some random colors
    # got this code from here: https://www.programcreek.com/python/?CodeExample=generate+colors
    colors = np.array(generate_colors(n_channels))
    print(colors)
    out_shape = list(img.shape)
    out_shape[2] = 3  ## change to RGB number of channels (3)
    out = np.zeros(out_shape)
    for chan in range(img.shape[2]):
        out = out + np.expand_dims(img[:, :, chan], axis=2) * np.expand_dims(colors[chan] / 255, axis=0)
    out = out / np.max(out)
    out = Image.fromarray((out * 255).astype(np.uint8))
    out.save(os.path.join(rgb_path, img_name[:8])+ '.png')

    plt.imshow(out)
    plt.axis('off')
    plt.show()

def generate_colors(class_names):
    hsv_tuples = [(x / class_names, 1., 1.) for x in range(class_names)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

# https://github.com/scikit-learn/scikit-learn
# Convert to 8bit images
def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, default=None
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, default=None
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, default=None
        Scale max value to `high`.  Default is 255.
    low : scalar, default=None
        Scale min value to `low`.  Default is 0.
    Returns

    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # list images
    img_list = os.listdir(tiff_path)
    for img_name in img_list:
        if img_name.endswith('tif'):
            # channels=>  0: ACTA2+, 1: Lineage tracing, 2: DAPI, 3: LGAL3+, 4: DIC
            img = io.imread(os.path.join(tiff_path, img_name))
            # convert to 8-bit
            img = bytescale(img)
            # calculate maximum intensity
            img = np.max(img, axis=0)
            img = np.moveaxis(img, 0, 2)
            # my proposal ordering channels:  put first Lineage tracing, then LGAL3+,DAPI ,ACTA2+
            # put np.zeros((img.shape[0], img.shape[1])) in case you do not have that marker

            # channels in the image =>  0: ACTA2+, 1: Lineage tracing, 2: DAPI, 3: LGAL3+, 4: DIC
            channel1 = img [:, :, 3]
            # channel2 = img [:, :, 3]
            channel2 = np.zeros((img.shape[0], img.shape[1]))
            channel3 = img[:, :, 2]
            # channel4 = img [:, :, 0]
            channel4 = np.zeros((img.shape[0], img.shape[1]))
            # create the histogram
            colors = ("red", "green", "blue", "pink")
            channel_ids = (0, 1, 2, 3)
            # create the histogram plot, with four lines, one for
            # each color
            plt.figure()
            plt.xlim([0, 256])
            for channel_id, c in zip(channel_ids, colors):
                histogram, bin_edges = np.histogram(
                img[:, :, channel_id], bins=256, range=(0, 256)
                )
                plt.plot(bin_edges[0:-1], histogram, color=c)

            plt.title("Color Histogram")
            plt.xlabel("Color value")
            plt.ylabel("Pixel count")
            plt.show()

            img = np.concatenate([channel1[np.newaxis, :], channel2[np.newaxis, :], channel3[np.newaxis, :], channel4[np.newaxis, :]])
            img = np.moveaxis(img, 0, 2)
            main(img, img_name)















