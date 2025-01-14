import cv2
import numpy as np

from segment_anything import SamPredictor, sam_model_registry


def initialize(model_type, ckpt_path):
    model = sam_model_registry[model_type](checkpoint=ckpt_path)
    predictor = SamPredictor(model)

    return predictor

def inference(input_path, prompt_bbox, predictor):
    input_img = cv2.imread(input_path)
    predictor.set_image(input_img)

    bbox = np.array(prompt_bbox)

    masks, scores, _ = predictor.predict(box=bbox, multimask_output=False)

    result_mask = masks[np.argmax(scores)]

    return result_mask

def visualize(input_img, input_mask, mask_color):
    overlay_mask = input_img.copy()
    overlay_mask[input_mask] = mask_color

    alpha = 0.5
    overlay_mask = cv2.addWeighted(input_img, 1 - alpha, overlay_mask, alpha, 0)

    return overlay_mask
