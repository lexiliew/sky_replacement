import cv2
from scipy.signal import medfilt
import numpy as np
from PIL import Image
import gradio as gr

def create_mask(mask):
    # assume the sky is in the upper part by default

    # Iterate through the columns of the mask to process each vertical slice
    for column_index in range(mask.shape[1]):
        column_values = mask[:, column_index]
        # Apply median filtering to the column
        after_median = medfilt(column_values, 19)
        try:
            # Find the index of the first zero and one in the filtered column
            first_zero_index = np.where(after_median == 0)[0][0]
            first_one_index = np.where(after_median == 1)[0][0]
            # Check if the distance between the first zero and first one is greater than 20
            if first_zero_index > 20:
                # Update the mask to separate the sky from other regions
                mask[first_one_index:first_zero_index, column_index] = 1
                mask[first_zero_index:, column_index] = 0
                mask[:first_one_index, column_index] = 0
        except:
            # Handle the case where no zero or one was found, continue to the next column
            continue
    return mask

def detect_sky_area(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise

    img_gray = cv2.blur(img_gray, (9, 3))
    # Apply median blur for smoothing
    img_gray = cv2.medianBlur(img_gray, 5)
    lap = cv2.Laplacian(img_gray, cv2.CV_8U)

    # Create a binary mask by thresholding the Laplacian gradient
    gradient_mask = (lap < 6).astype(np.uint8)
    # Define a morphological kernel for erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    # Apply morphological erosion to refine the gradient mask
    mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_ERODE, kernel)
    # Create a mask to separate the sky region from other areas
    mask = create_mask(mask)
    after_img = cv2.bitwise_and(img, img, mask=mask)
    return after_img

def replace_sky_with_angel_corrected(target_img, angel_img_path):
    angel_image = cv2.imread(angel_img_path)
    target_output_image = detect_sky_area(target_img)
    sky_mask = target_output_image[:, :, 0] == 0
    mask_single_channel = (sky_mask * 255).astype(np.uint8)
    resized_angel = cv2.resize(angel_image, (target_img.shape[1], target_img.shape[0]), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.bitwise_not(mask_single_channel)
    angel_region = cv2.bitwise_and(resized_angel, resized_angel, mask=mask_inv)
    non_sky_region = cv2.bitwise_and(target_img, target_img, mask=mask_single_channel)
    replaced_img = cv2.add(non_sky_region, angel_region)
    return cv2.cvtColor(replaced_img, cv2.COLOR_BGR2RGB)  # Convert to RGB for Gradio

def pil_to_cv(pil_image):
    numpy_image = np.array(pil_image)
    return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

def process_image(target_image_pil):
    target_image = pil_to_cv(target_image_pil)
    replaced_sky_image_rgb = replace_sky_with_angel_corrected(target_image, 'angel.jpg')  # Path to the default angel image
    return Image.fromarray(replaced_sky_image_rgb)

iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(),
    outputs=gr.Image(type="pil"),
    title="Sky Replacement App",
    description="Upload a target image to replace the sky area with a default angel image."
)

iface.launch()
