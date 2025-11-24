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
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageChops
import clip
from IPython.display import display
from sklearn.cluster import KMeans
robot_params = RobotParams()
env = DiffBotEnv(gui=True, n_objects=15, seed=None, robot_params=robot_params)

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

def plot_sector(comp):
    if not hasattr(plot_sector, "_im"):
        plt.ion()
        plot_sector._fig, plot_sector._ax = plt.subplots()
        plot_sector._im = plot_sector._ax.imshow(np.array(comp))
        plot_sector._ax.axis("off")
        plt.show(block=False)
    else:
        # Update existing image
        plot_sector._im.set_data(np.array(comp))
        plot_sector._fig.canvas.draw()



import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

def clip_heatmap(image_pil, txt_emb, model, preprocess, device):
    # Preprocess image
    img_in = preprocess(image_pil).unsqueeze(0).to(device)
    img_in.requires_grad_(True)
    
    # Preprocess text
    txt_emb_norm = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

    # Forward pass to get image embedding
    img_emb = model.encode_image(img_in)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    # Similarity
    sim = (img_emb @ txt_emb_norm.T)
    
    # Backpropagate
    sim.backward()

    # Get gradients of the image
    grads = img_in.grad[0]              # (3,H,W)
    grads = grads.mean(dim=0).cpu().numpy()

    # Resize grads to image resolution
    grads = (grads - grads.min()) / (grads.max() - grads.min() + 1e-6)
    heatmap = Image.fromarray((grads * 255).astype(np.uint8)).resize(image_pil.size)

    return heatmap, sim.item()


def score_segments(img, text_emb):
    """
    Segment the image using SAM, return CLIP cosine scores per segment.
    """
    img_np = np.array(img)
    
    # segment using heatmap clustering
    heatmap, sim_items = clip_heatmap(img, text_emb, clip_model, clip_preprocess, device)
    heatmap_np = np.array(heatmap).astype(np.float32) / 255.0
    H, W = heatmap_np.shape
    n_segments = 7
    X = heatmap_np.reshape(-1, 1)  # (H*W, 1)
    kmeans = KMeans(n_clusters=n_segments, random_state=0).fit(X)
    labels = kmeans.labels_.reshape(H, W)  # (H, W)
    masks = [(labels == i).astype(np.uint8) for i in range(n_segments)] 

    

    crops = []
    for mask in masks:
        # extract segment
        comp = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
        segment = ImageChops.multiply(img, comp.convert("RGB"))
        crops.append(clip_preprocess(segment))
    imgs = torch.stack(crops).to(device)

    with torch.no_grad():
        img_feats = clip_model.encode_image(imgs)
        #(img_feats.shape)  #
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)  # [N, d]
        sims = (img_feats @ text_emb.T).squeeze(1)                    # [N]
    return sims, masks



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


GOAL_SIM_THRESHOLD = 0.27
GOAL_FRAMES_REQUIRED = 20    # number of consecutive frames before stopping
_goal_counter = 0
old_sims = None

def object_too_close(sims):
    sims_np = sims.cpu().numpy()
    s_max = sims_np.max()
    s_second = np.partition(sims_np, -2)[-2]
    return (s_max - s_second) > 0.25


def goal_reached(sims):
    global _goal_counter, old_sims
    max_sim = float(torch.max(sims).item())

    # # id sudden drop in score object if very close 
    if old_sims is not None:
        if max_sim < GOAL_SIM_THRESHOLD and old_sims - max_sim > 0.02:
            return True

    # if score above threshold we increment counter
    if max_sim >= GOAL_SIM_THRESHOLD:
        _goal_counter += 1
    else:
        _goal_counter = 0

    old_sims = max_sim

    # if counter is above required frames we stop
    if _goal_counter >= GOAL_FRAMES_REQUIRED:
        _goal_counter = 0
        return True

    return False


def _sector_to_cmd(sims, K, fov_deg=90.0, v_max=15.00, w_gain=7.5, conf_th=0.08):
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

SMOOTH_v = 0.15
SMOOTH_w = 0.03
K_SECTORS = 5

def _seg_to_cmd(sims, masks, fov_deg=120.0, v_max=15.0, w_gain=7.5, conf_th=0.08):
    """
    Computes command from segment scores.
    """
    N = len(sims)
    idx = int(torch.argmax(sims).item())
    fov = math.radians(fov_deg)
    # center angles for segments
    bearings = np.linspace(-fov/2, fov/2, N)
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

def compute_next_cmd(v, w, camera_feed, prompt):
    global text_emb, _prev_v, _prev_w
    if camera_feed is None:
        return 0.0, 0.0 # stop the robot if no camera feed

    text_emb = _encode_text(prompt)  # text is global for efficiencycompute_next_cmd

    #converts the image and computes the sector scores
    pil = _to_pil(camera_feed)
    sims, masks = _score_sectors(pil, text_emb, K=K_SECTORS)
    #sims, masks = score_segments(pil, text_emb)
    #print(sims)  #

    if goal_reached(sims):
        print("\n ---------- GOAL REACHED! ---------")
        return 0.0, 0.0, sims, masks

    # Convert chosen sector to bearing & forward speed
    #spin if no positive matches
    if torch.all(torch.abs(sims - sims[0]) < 0.03):
        # no positive matches
        if torch.all(sims < 0.1):
            v_cmd, w_cmd, idx, conf = 0.0, 0.0, -1, 0.0
        else:
            v_cmd, w_cmd, idx, conf = 0.0, 0.6, -1, 0.0
    else:
        v_cmd, w_cmd, idx, conf = _sector_to_cmd(sims, K=K_SECTORS, fov_deg=120.0)
        #v_cmd, w_cmd, idx, conf = _seg_to_cmd(sims, masks, fov_deg=120.0)
    print(f"Chosen sector: {idx}, v_cmd: {v_cmd:.2f}, w_cmd: {w_cmd:.2f}, conf: {conf:.2f}")

    # Smooth commands
    v_sm = SMOOTH_v * _prev_v + (1.0 - SMOOTH_v) * v_cmd
    w_sm = SMOOTH_w * _prev_w + (1.0 - SMOOTH_w) * w_cmd
    _prev_v, _prev_w = v_sm, w_sm
    return float(v_sm), float(w_sm), sims, masks

def plot_camera_feed(frame_bgr, sims=None, masks=None):
    rgb = frame_bgr[:, :, ::-1]
    pil = Image.fromarray(rgb)

    # Draw sector lines and scores
    if sims is not None and masks is None :
        W, H = pil.size
        w = W // K_SECTORS
        draw = ImageDraw.Draw(pil)
        for i in range(K_SECTORS):
            x = i * w
            draw.line([(x, 0), (x, H)], fill=(0, 255, 0), width=2)
            draw.text((x + 5, 5), f"{sims[i].item():.2f}", fill=(255, 0, 0))

    if masks is not None:
        draw = ImageDraw.Draw(pil)
        for i, mask in enumerate(masks):
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            # Create an RGBA image for the mask
            mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
            colored_mask = Image.new("RGBA", pil.size, color + (0,))
            colored_mask.putalpha(mask_img)
            pil = Image.alpha_composite(pil.convert("RGBA"), colored_mask)
            # Draw score
            draw.text((10, 10 + i * 20), f"{sims[i].item():.2f}", fill=color)
    

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
        v,w, sims, masks = compute_next_cmd(v, w, camera_feed, prompt)
        plot_camera_feed(camera_feed, sims)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
        env.reset_pose()
        p.disconnect()
