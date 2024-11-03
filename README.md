# Interactive SAM

I created this project because I need to compare performance between [SAM]((https://github.com/facebookresearch/segment-anything)), [SAM-HQ]((https://github.com/SysCV/sam-hq)) and SAM2 for image segmentation. I decided to share this in case someone might want find it helpful

### Supported type of input
- __Brush__: Draw an area on the object that you want to segment, it will uniformly sample 10 points (this number can be changed) inside that area and use ihem as positive points
- __Box__: Draw a bounding box around an object to segment (this work similar to LASSO tool in photoshop but since Gradio does not support LASSO, I use a brush as replacement to draw)
- __Points__: choose which point on the image as positive and negative points to segment in SAM

### Installation
- Install pytorch
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```
- Then follow the instruction from each repo to install [segment-anything](https://github.com/facebookresearch/segment-anything), [sam-hq](https://github.com/SysCV/sam-hq), [sam2](https://github.com/facebookresearch/sam2)
- Download checkpoints in `checkpoints` folder and run gradio `python gradio.py` and enjoy
### Some examples