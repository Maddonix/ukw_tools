import albumentations as A
import numpy as np


def crop_img(img, crop):
    # crop is: ymin, ymax, xmin, xmax
    ymin, ymax, xmin, xmax = crop
    img = img[ymin:ymax, xmin:xmax, :]

    y, x, _ = img.shape
    delta = x - y
    if delta > 0:
        _padding = [(abs(delta), 0), (0, 0), (0, 0)]
        img = np.pad(img, _padding)
    elif delta < 0:
        _padding = [(0, 0), (abs(delta), 0), (0, 0)]
        img = np.pad(img, _padding)

    return img


img_augmentations = A.Compose(
    [
        # Rotates the image 90, 180 or 270 degrees
        A.RandomRotate90(p=0.5),
        # Flips the image vertically, horizontally or both
        A.Flip(p=0.5),
        # Interchanges x and y dimension of the image
        A.Transpose(p=0.5),
        # Applies a gaussian noise filter with a variability limit of 10 - 50 and mean of 0
        A.GaussNoise(p=0.2),
        # Applies one of the included blur algorithms
        A.OneOf(
            [
                A.MotionBlur(),
                A.MedianBlur(blur_limit=3),
                A.Blur(blur_limit=3),
            ],
            p=0.2,
        ),
        # Randomly applies one of: shift image on x or y axis, rescale image, rotate image
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.1),
        # Randomly appplies one of the distortion algorithm
        A.OneOf(
            [
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.augmentations.PiecewiseAffine(p=0.3),
            ],
            p=0.2,
        ),
        A.OneOf(
            [
                # Apply contrast limited adaptive histogram equalization
                A.CLAHE(clip_limit=2),
                # Increases image sharpness and overlays it with the original image
                A.Sharpen(),
                # Replaces pixels by highliths and shadows and overlays it with the original image
                A.Emboss(),
                # Applies random brightness and contrast values
                A.RandomBrightnessContrast(),
            ],
            p=0.3,
        ),
        # Randomly shift hue and saturation of the image
        # A.HueSaturationValue(p=0.3),
        # Randomly cut out an image section
        # A.Cutout(p = 0.1)
    ]
)

img_transforms = A.Compose(
    [
        A.Normalize(
            mean=(0.45211223, 0.27139644, 0.19264949),
            std=(0.31418097, 0.21088019, 0.16059452),
            max_pixel_value=255,
        )
    ]
)
