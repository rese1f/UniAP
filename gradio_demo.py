import torch

import os
import yaml
import gradio as gr
import numpy as np
from easydict import EasyDict
from einops import rearrange

from PIL import Image
import matplotlib.pyplot as plt

from train.train_utils import load_model


def generate_gaussian_heatmap(heatmap):
    # Find the position of the maximum value in the heatmap
    max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    # Create a grid with coordinates
    y, x = np.mgrid[0:heatmap.shape[0], 0:heatmap.shape[1]]

    # Compute the Gaussian distribution centered at the max_pos
    sigma = 5  # You can adjust the spread of the Gaussian distribution by changing this value
    gaussian_heatmap = np.exp(-((x - max_pos[1])**2 + (y - max_pos[0])**2) / (2 * sigma**2))

    # Normalize the Gaussian heatmap to have values between 0 and 1
    gaussian_heatmap = gaussian_heatmap / np.max(gaussian_heatmap)

    return gaussian_heatmap

def vis_heatmap(img, prediction):
    # Convert the torch tensor to a numpy array and squeeze the dimensions
    image_np = img.squeeze().permute(1, 2, 0).numpy()
    heatmap_np = prediction.reshape((224, 224, 1))
    # Reshape the heatmap to match the image dimensions
    # heatmap_resized = np.repeat(heatmap_np, 3, axis=2)  # Repeat along channel dimension

    plt.imshow(image_np)
    plt.imshow(heatmap_np, alpha=0.5, cmap='viridis')  # Adjust the alpha and cmap as needed
    plt.axis('off')  # Turn off axis labels

    # Convert the Matplotlib figure to a PIL image
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust the subplot layout
    plt.gcf().canvas.draw()  # Draw the canvas
    pil_image = Image.frombytes('RGB', plt.gcf().canvas.get_width_height(), plt.gcf().canvas.tostring_rgb())
    
    return pil_image

class Demo:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            self.config = EasyDict(self.config)
        self.model, _ = load_model(self.config, verbose=True)

    def preprocess(self, *args):
        imgs = []
        for img in args:
            img = img.resize((224, 224))
            img = np.asarray(img)
            # type conversion
            img = img.astype('float32') / 255
            # shape conversion
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img)[None, None, None, :]
            imgs.append(img)
        return imgs

    def forward(self, support_image, support_label, query_image):
        # pre-processing
        support_image, support_label, query_image = self.preprocess(support_image, support_label, query_image)
        support_label = support_label[:, :, :, 0:1]
        t_idx = torch.tensor([[self.config.task_id]])

        # model forward
        prediction = self.model(support_image, 
                                support_label, 
                                query_image, 
                                t_idx=t_idx, 
                                sigmoid=True)
        
        # post-processing

        prediction = rearrange(prediction, 'B T N C H W -> (B T N C) H W')
        prediction = generate_gaussian_heatmap(prediction[0].detach().cpu().numpy())
        # prediction = (prediction > 0.2).float()
        # prediction = prediction[0].detach().cpu().numpy()
        vis_prompt = vis_heatmap(support_image, np.squeeze(support_label))
        vis_prediction = vis_heatmap(query_image, prediction)
        
        return vis_prompt, vis_prediction, prediction
    
    # def calculate_pck(self):
    #     return

    # def calculate_acc(self):
    #     return




if __name__ == "__main__":
    demo = Demo(
        config_path="configs/demo.yaml",
    )

    gr.Interface(
        fn=demo.forward,
        inputs=[
            gr.Image(source='upload', type="pil", label="support image"),
            gr.Image(source='upload', type="pil", label="support label"),
            gr.Image(source='upload', type="pil", label="query image"),
        ],
        outputs=[
            gr.outputs.Image(type="numpy", label="vis_support"),
            gr.outputs.Image(type="numpy", label="vis_prediction"),
            gr.outputs.Image(type="numpy", label="htm_prediction"),
        ],
        examples=[
            [
                os.path.join(os.path.dirname(__file__), "assets/vis_set1/imgid_4534_rgb.png"),
                os.path.join(os.path.dirname(__file__), "assets/vis_set1/point_4534_view_2_heatmap.png"),
                os.path.join(os.path.dirname(__file__), "assets/vis_set1/imgid_14836_rgb.png"),
            ],
        ],
        allow_flagging='never',
        cache_examples=False,
    ).launch()