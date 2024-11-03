echo "Downloading SAM..."
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

echo "Downloading SAM2..."
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

echo "Downloading SAM-HQ"
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth?download=true -O sam_hq_vit_b.pth
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth?download=true -O sam_hq_vit_h.pth
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth?download=true -O sam_hq_vit_l.pth
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_tiny.pth?download=true -O sam_hq_vit_tiny.pth