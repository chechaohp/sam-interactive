import os
os.environ["OPENBLAS_NUM_THREADS"] = "16"
import cv2
import torch
import logging
import numpy as np
import gradio as gr

from typing import Tuple, Dict, Any, List
from skimage import color
from segment_anything import (
    SamAutomaticMaskGenerator,
    build_sam_vit_b,
    build_sam_vit_h,
    build_sam_vit_l,
    SamPredictor,
)

import cv2

from gradio.themes import Monochrome

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from segment_anything_hq import sam_model_registry
from segment_anything_hq import SamPredictor as SamPredictorHQ

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SAM_CHECKPOINT_PATH = "checkpoints"
SAM2_CHECKPOINT_PATH = "checkpoints"
SAM_HQ_CHECKPOINT_PATH = "checkpoints"

USE_SAM = True
USE_SAM2 = True
USE_SAM_HQ = True

def load_model(
    name: str,
    device: str,
) -> SamPredictor:
    if 'vit' in name and not name.startswith('sam_hq'):
        # for SAM 1
        checkpoint_path = os.path.join(SAM_CHECKPOINT_PATH, name)
        if "vit_b" in name:
            model = build_sam_vit_b(checkpoint_path)
        elif "vit_h" in name:
            model = build_sam_vit_h(checkpoint_path)
        elif "vit_l" in name:
            model = build_sam_vit_l(checkpoint_path)
        else:
            raise ValueError(f"Invalid checkpoint name: {name}")
        version = 1
    elif name.startswith('sam2'):
        checkpoint_path = os.path.join(SAM2_CHECKPOINT_PATH,name)
        if name == 'sam2.1_hiera_base_plus.pt':
            config_file = 'sam2.1_hiera_b+.yaml'
        elif name == 'sam2.1_hiera_large.pt':
            config_file = 'sam2.1_hiera_l.yaml'
        elif name == 'sam2.1_hiera_small.pt':
            config_file = 'sam2.1_hiera_s.yaml'
        elif name == 'sam2.1_hiera_tiny.pt':
            config_file = 'sam2.1_hiera_t.yaml'
        elif name == 'sam2_hiera_base_plus.pt':
            config_file = 'sam2_hiera_b+.yaml'
        elif name == 'sam2_hiera_large.pt':
            config_file = 'sam2_hiera_l.yaml'
        elif name == 'sam2_hiera_small.pt':
            config_file = 'sam2_hiera_s.yaml'
        elif name == 'sam2_hiera_tiny.pt':
            config_file = 'sam2_hiera_t.yaml'
        else:
            raise ValueError(f"cannot find {name}")
        model = build_sam2(config_file, checkpoint_path)
        version = 2
    elif name.startswith('sam_hq'):
        model_type = name.replace('sam_hq_','')
        checkpoint_path = os.path.join(SAM_HQ_CHECKPOINT_PATH,name+'.pth') 
        model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        version = 3
    else:
        raise ValueError(f'cannot find mode {name}')

    model.to(device)
    logger.info(f"Loaded model: {name}")
    if version == 1:
        return SamPredictor(model)
    elif version == 2:
        return SAM2ImagePredictor(model)
    elif version == 3:
        return SamPredictorHQ(model)
    else:
        raise NotImplementedError()

def display_detected_keypoints(
    image: np.ndarray,
    brush: np.ndarray,
    bbox: np.ndarray,
    positive_points: np.ndarray,
    negative_points: np.ndarray,
    num_brush_sample: int,
    progress = gr.Progress()
):
    logger.info("Detecting query information")
    sections = []
    # detect boxes
    if type(bbox) == dict:
        bbox = bbox["composite"][:,:,:3]
    assert bbox.shape == image.shape, "Canvas shape should be the same as input image"
    box,diff_mask = detect_boxes(image, bbox)
    diff_mask[diff_mask>0] = 1
    if box is not None:
        sections.append((tuple(box), "Bounding Box"))
    progress(0.25, "Detecting bounding boxes", total=4)
    # detect brush
    if type(brush) == dict:
        brush = brush["composite"][:,:,:3]
    assert brush.shape == image.shape, "Canvas shape should be the same as input image"
    brush_sample, _ , _ = detect_brush(image, brush, num_brush_sample)
    if brush_sample is not None:
        sections.append((brush_sample, "Brush Sample Points"))
    progress(0.5, "Detecting brush", total=4)
    
    # detect positive points
    if type(positive_points) == dict:
        positive_points = positive_points["composite"][:,:,:3]
    assert positive_points.shape == image.shape, "Canvas shape should be the same as input image"
    pos_points, _ , _ = detect_brush(image, positive_points, None, False)
    if pos_points is not None:
        sections.append((pos_points, "Positive Points"))
    progress(0.75, "Detecting positive points", total=4)
    
    # detect negative points
    if type(negative_points) == dict:
        negative_points = negative_points["composite"][:,:,:3]
    assert negative_points.shape == image.shape, "Canvas shape should be the same as input image"
    neg_points, _ , _ = detect_brush(image, negative_points, None, False)
    if pos_points is not None:
        sections.append((neg_points, "Positive Points"))

    progress(1, "Detecting negative points", total=4)
    return (image, sections)

def process_drawing_mask(diff_mask:np.ndarray, mode = 'erode'):
    assert mode in ["erode","dilate"]
    kernel = np.ones((5,5),np.uint8)
    diff_mask = diff_mask.astype(np.uint8)
    if mode == "erode":
        return cv2.erode(diff_mask,kernel,iterations = 3)
    else:
        return cv2.dilate(diff_mask, kernel, iterations = 3)
    

def detect_boxes(image: np.ndarray, image_with_boxes: np.ndarray) -> np.ndarray:
    if type(image_with_boxes) == dict:
        image_with_boxes = image_with_boxes["composite"][:,:,:3]

    diff_mask = np.sum(image != image_with_boxes, axis=-1) > 0
    if np.max(diff_mask) == 0:
        return None, None
    diff_mask = process_drawing_mask(diff_mask)
    bounding_mask = diff_mask.astype(np.uint8)
    x, y, w, h = cv2.boundingRect(bounding_mask)
    box = np.array([x, y, x + w, y + h])
    return box, diff_mask

def detect_brush(image: np.ndarray, image_with_brush: np.ndarray, num_brush_sample: int = None, process_mask = True) -> np.ndarray:
    if type(image_with_brush) == dict:
        image_with_brush = image_with_brush["composite"][:,:,:3]

    diff_mask = np.sum(image != image_with_brush, axis=-1) > 0
    if np.max(diff_mask) == 0:
        return None, None, None
    if process_mask:
        diff_mask = process_drawing_mask(diff_mask)
    else:
        diff_mask = diff_mask.astype(np.uint8)

    ind = np.nonzero(diff_mask.squeeze())
    indexes = np.array(list(zip(ind[0],ind[1])))
    if num_brush_sample is not None:
        np.random.shuffle(indexes)
        indexes = indexes[:num_brush_sample]
    brush_sample = np.zeros_like(diff_mask)
    brush_sample[indexes[:,0],indexes[:,1]] = 1
    brush_sample = process_drawing_mask(brush_sample, mode='dilate')
    return brush_sample, indexes, diff_mask

index = 0

@torch.no_grad()
def guided_prediction(
    predictor: SamPredictor,
    image: np.ndarray,
    low_res_mask:np.ndarray,
    box_canvas: np.ndarray,
    brush_canvas: np.ndarray,
    positive_points: np.ndarray,
    negative_points: np.ndarray,
    num_brush_sample: int = None,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[np.ndarray, np.ndarray]:
    global index
    logger.info("start predicting")
    progress(0.5, "Detecting Query", total=4)
    if type(box_canvas) == dict:
        bbox = box_canvas["composite"][:,:,:3]
    assert bbox.shape == image.shape, "Canvas shape should be the same as input image"
    box, _ = detect_boxes(image, bbox)
    # detect brush
    if type(brush_canvas) == dict:
        brush = brush_canvas["composite"][:,:,:3]
    assert brush.shape == image.shape, "Canvas shape should be the same as input image"
    _, brush_pts , _ = detect_brush(image, brush, num_brush_sample)
    
    # detect positive points
    if type(positive_points) == dict:
        positive_points = positive_points["composite"][:,:,:3]
    assert positive_points.shape == image.shape, "Canvas shape should be the same as input image"
    _ , pos_pts , _ = detect_brush(image, positive_points, None, False)
    
    # detect negative points
    if type(negative_points) == dict:
        negative_points = negative_points["composite"][:,:,:3]
    assert negative_points.shape == image.shape, "Canvas shape should be the same as input image"
    _ , neg_pts , _ = detect_brush(image, negative_points, None, False)
    
    sam_input = {"multimask_output": True}
    if box is not None:
        sam_input["box"] = box
    input_pts = []
    input_label = []
    if brush_pts is not None:
        input_pts.append(brush_pts)
        input_label.extend([1]*len(brush_pts))
    if pos_pts is not None:
        input_pts.append(pos_pts)
        input_label.extend([1]*len(pos_pts))
    if neg_pts is not None:
        input_pts.append(neg_pts)
        input_label.extend([0]*len(neg_pts))
    
    if len(input_pts) > 0:
        input_pts = np.vstack(input_pts)[:,[1,0]]
        input_label = np.array(input_label)
        sam_input["point_coords"] = input_pts
        sam_input["point_labels"] = input_label

    progress(0.75, "Predicting masks", total=4)
    masks, scores, low_res_mask = predictor.predict(**sam_input)
    # Mask input is 1 x H x W
    # low_res_mask = low_res_mask[[np.argmax(scores)]]

    masks = masks.astype(int)
    colors = [[(1, 0, 0)], [(0, 1, 0)], [(0, 0, 1)]]
    if len(masks) == 1:
        color_masks = [color.label2rgb(masks[0], image, colors[0])]
    else:
        color_masks = [color.label2rgb(masks[i], image, c) for i, c in enumerate(colors)]
    print('mask scores', scores)
    progress(1, "Returning masks", total=4)
    low_res_mask = low_res_mask[[np.argmax(scores)]]

    return color_masks, low_res_mask, scores


def set_sketch_images(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger.info(f'Input mage shape {image.shape}')
    return image, image, image, image


def compute_image_embedding(predictor: SamPredictor, image: np.ndarray):
    predictor.set_image(image)
    logger.info('Done compute image embedding')  
    # Reset best mask
    return None


device = "cuda" if torch.cuda.is_available() else "cpu"
print('Using ',device)
assert USE_SAM or USE_SAM2 or USE_SAM_HQ, "Need to use atleast one of the SAM model"
available_models = []

# add sam 1
if USE_SAM:
    for x in os.listdir(SAM_CHECKPOINT_PATH):
        if x.endswith(".pth"):
            available_models.append(x)
# add sam 2
if USE_SAM2:
    for x in os.listdir(SAM2_CHECKPOINT_PATH):
        if x.endswith(".pt"):
            available_models.append(x)

# add sam-hq
if USE_SAM_HQ:
    sam_hq_model_type = ["sam_hq_vit_l","sam_hq_vit_b","sam_hq_vit_h","sam_hq_vit_tiny"]
    available_models.extend(sam_hq_model_type)


# set default value for choosing model
default_model = available_models[0]
default_predictor = load_model(default_model, device='cuda')

with gr.Blocks(theme=gr.themes.Citrus()) as application:
    gr.Markdown(value="""# Segment Anything At Home
                Support SAM, SAM2 and SAM-HQ

                - Brush Canvas to sample positive point inside brush
                
                - Box Canvas to sample points on the drawing

                - Positive points to draw positive points

                - Negative Points to draw negative points

                __Note__: only SAM and SAM2 return multiple masks
                """)
    with gr.Tab("Detetion"):
        selected_model = gr.Dropdown(choices=available_models, label="Model",
                                    value=default_model, interactive=True,)

        predictor_state = gr.State(default_predictor)
        selected_model.change(lambda x: load_model(x, device), inputs=[
                                selected_model], outputs=[predictor_state], show_progress=True)

        with gr.Row():
            with gr.Column(scale=0):
                base_image = gr.Image(label="Base Image", sources="upload",scale=0)
                num_brush_sample = gr.Number(label="Number of Brush Sample",value=10,interactive=True)

            with gr.Column():
                with gr.Row():
                    brush_canvas = gr.ImageEditor(label="Brush Canvas",
                                                sources=('upload'), 
                                                brush=gr.Brush(default_size=10,
                                                               default_color=["#000000"]),
                                                interactive=True,
                                                container=True,
                                                height="auto",
                                                format='png',scale=1)
                    box_canvas = gr.ImageEditor(label="Box Canvas",
                                                sources=('upload'), 
                                                brush=gr.Brush(default_size=10,
                                                               default_color=["#000000"]),
                                                interactive=True,
                                                container=True,
                                                height="auto",
                                                format='png',scale=1)
                with gr.Row():
                    pos_pts = gr.ImageEditor(label="Positive Points",
                                            sources=("upload"), 
                                            brush=gr.Brush(default_size=2,default_color=["#f44336"],color_mode="fixed"),
                                            interactive=True,
                                            container=True,
                                            height="auto",
                                            format='png',
                                            scale=1)
                    neg_pts = gr.ImageEditor(label="Negative Points",
                                            sources=("upload"), 
                                            brush=gr.Brush(default_size=2,default_color=["#0fff00"],color_mode="fixed"),
                                            interactive=True,
                                            container=True,
                                            height="auto",
                                            format='png',
                                            scale=1)

        with gr.Row():
            annotated_canvas = gr.AnnotatedImage(label="Annotated Canvas",color_map={"Positive": "#46ff33", "Negative": "#ff3333", "Bounding Box": "#3361ff"})
            output_masks = gr.Gallery(label="Output Masks",preview=True)

        with gr.Row():
            get_query = gr.Button("Visual Query")
            compute_embed = gr.Button("Compute Embedding")
            predict = gr.Button("Predict")

        predict_score = gr.Text(label='Mask score', value = "score")

        previous_mask = gr.State()
        base_image.upload(compute_image_embedding, inputs=[predictor_state, base_image], outputs=previous_mask)
        base_image.upload(set_sketch_images, inputs=[base_image], outputs=[brush_canvas,box_canvas, pos_pts, neg_pts])
        

        get_query.click(display_detected_keypoints,
                inputs=[base_image, brush_canvas, box_canvas, pos_pts, neg_pts, num_brush_sample],
                outputs=[annotated_canvas],)
        
        compute_embed.click(compute_image_embedding, inputs=[predictor_state, base_image], outputs=previous_mask)
        
        predict.click(guided_prediction,
                    inputs=[predictor_state, base_image, previous_mask, box_canvas, brush_canvas, pos_pts, neg_pts,num_brush_sample],
                    outputs=[output_masks, previous_mask, predict_score])

    with gr.Tab("Segment Everything"):
        text = gr.Text("TODO")

application.queue()
application.launch()
