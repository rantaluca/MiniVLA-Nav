## Imports
import argparse
import math
import random
import os
import time
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pybullet as p
import pybullet_data
from env import DiffBotEnv, RobotParams
import matplotlib.pyplot as plt
import torch
from PIL import Image
import clip
from IPython.display import display
from PIL import Image  
from PIL import ImageDraw 


robot_params = RobotParams()
env = DiffBotEnv(gui=True, n_objects=15, seed=185, robot_params=robot_params)

# Load CLIP model
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def _to_pil(img_np):
    # img_np: HxWx3 uint8 from env.step(camera_feed=True) BVR
    # convert BGR to RGB
    img_np = img_np[:, :, ::-1]
    return Image.fromarray(img_np)

def _encode_text(prompt):
    with torch.no_grad():
        tokens = clip.tokenize([prompt]).to(device)
        txt = clip_model.encode_text(tokens)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt  # [1, d]

def _score_sectors(img, text_emb, K=7):
    """
    Split the image into K vertical sectors, return CLIP cosine scores per sector.
    """
    W, H = img.size
    w = W // K
    crops = []
    boxes = []
    for i in range(K):
        x0 = max(0, i*w - w//4)                 # small overlap
        x1 = min(W, x0 + w + w//2)
        crop = img.crop((x0, 0, x1, H))
        crops.append(clip_preprocess(crop))
        boxes.append((x0, x1))
    imgs = torch.stack(crops).to(device)

    with torch.no_grad():
        img_feats = clip_model.encode_image(imgs)
        #(img_feats.shape)  #
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)  # [K, d]
        sims = (img_feats @ text_emb.T).squeeze(1)                    # [K]
    return sims, boxes

def _sector_to_cmd(sims, K, fov_deg=120.0, v_max=10.00, w_gain=3.5, conf_th=0.12):
    """
    sims: [K] cosine scores. Pick argmax -> bearing in [-FOV/2, +FOV/2] (radians).
    Forward speed scales with confidence and reduces when turning sharply.
    """
    K = int(K)
    idx = int(torch.argmax(sims).item())
    fov = math.radians(fov_deg)
    # center angles for sectors
    bearings = np.linspace(-fov/2, fov/2, K)
    bearing = float(bearings[idx])
    conf = float(torch.softmax(sims, dim=0)[idx])
    # turn rate
    w = - w_gain * bearing

    # forward speed: zero if low confidence; zotherwise slower when turning
    if conf < conf_th:
        v = 0.1
    else:
        v = v_max * conf * max(0.0, math.cos(bearing)) 
    return v, w, idx, conf

SMOOTH_v = 0.5
SMOOTH_w = 0.15
K_SECTORS = 9

def compute_next_cmd(v, w, camera_feed, prompt):
    global text_emb, _prev_v, _prev_w
    if camera_feed is None:
        return 0.0, 0.0 # stop the robot if no camera feed

    text_emb = _encode_text(prompt)  # text is global for efficiency

    #converts the image and computes the sector scores
    pil = _to_pil(camera_feed)
    sims, _ = _score_sectors(pil, text_emb, K=K_SECTORS)
    #print(sims)  #

    # Convert chosen sector to bearing & forward speed
    #stop if all similarity scores are equal
    if torch.all(torch.abs(sims - sims[0]) < 0.03):
        # no positive matches
        v_cmd, w_cmd, idx, conf = 0.0, 0.0, -1, 0.0
    else:
        v_cmd, w_cmd, idx, conf = _sector_to_cmd(sims, K=K_SECTORS, fov_deg=90.0)
    print(f"Chosen sector: {idx}, v_cmd: {v_cmd:.2f}, w_cmd: {w_cmd:.2f}, conf: {conf:.2f}")

    # Smooth commands
    v_sm = SMOOTH_v * _prev_v + (1.0 - SMOOTH_v) * v_cmd
    w_sm = SMOOTH_w * _prev_w + (1.0 - SMOOTH_w) * w_cmd
    _prev_v, _prev_w = v_sm, w_sm
    return float(v_sm), float(w_sm), sims

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def plot_camera_feed(frame_bgr, sims=None):
    rgb = frame_bgr[:, :, ::-1]
    pil = Image.fromarray(rgb)

    # Draw sector lines and scores
    if sims is not None:
        W, H = pil.size
        w = W // K_SECTORS
        draw = ImageDraw.Draw(pil)
        for i in range(K_SECTORS):
            x = i * w
            draw.line([(x, 0), (x, H)], fill=(0, 255, 0), width=2)
            draw.text((x + 5, 5), f"{sims[i].item():.2f}", fill=(255, 0, 0))

    # Initialize live window
    if not hasattr(plot_camera_feed, "_im"):
        plt.ion()
        plot_camera_feed._fig, plot_camera_feed._ax = plt.subplots()
        plot_camera_feed._im = plot_camera_feed._ax.imshow(np.array(pil))
        plot_camera_feed._ax.axis("off")
        plt.show(block=False)
    else:
        # Update existing image
        plot_camera_feed._im.set_data(np.array(pil))
        plot_camera_feed._fig.canvas.draw()


def main():
    global v, w
    v, w = 0.0, 0.0
    global _prev_v, _prev_w
    _prev_v, _prev_w = 0.0, 0.0
    prompt = input("Enter target object description: ")
    #prompt = "a photo of a small bright red ball, red sphere, vivid red color"
    #prompt = "a small spherical cyan ball next to a bigger red ball"
    while True:
        camera_feed = env.step(steps=30, cmd=(v, w), camera_feed=True)
        # plot the camera feed
        v,w, sims = compute_next_cmd(v, w, camera_feed, prompt)
        plot_camera_feed(camera_feed, sims)
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
        env.reset_pose()
        p.disconnect()
