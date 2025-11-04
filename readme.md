# MiniVLA-NAV

<div align="center">
  <h2>Mini Vision-Language-Action Navigation with CLIP and PyBullet</h2>
    <video width="600" controls>
  <source src="demos/coffee.mp4" type="video/mp4">
</video>
<h4>Given prompt: "an object useful to drink coffee"</h4>

</div>
<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-≥3.10-blue.svg?logo=python&logoColor=white" alt="Python Version">
  </a>
  <a href="https://pybullet.org/">
    <img src="https://img.shields.io/badge/PyBullet-Physics%20Engine-red.svg?logo=python&logoColor=white" alt="PyBullet">
  </a>
  <a href="https://github.com/openai/CLIP">
    <img src="https://img.shields.io/badge/CLIP-Vision--Language--Model-purple.svg?logo=openai&logoColor=white" alt="CLIP">
  </a>
</p>

---

## Overview

**MiniVLA-NAV** is a **Mini Vision-Language-Action (VLA)** project I created to start experimenting with embodied navigation guided by a **Vision-Language Model (VLM)**.  
The system uses **OpenAI CLIP (ViT-B/32)** to connect text and perception: the robot receives a natural-language prompt, analyzes its camera feed, and navigates toward the region that best matches the description.  

The entire environment — robot, world, textures, and objects — is built procedurally in **PyBullet**.  
The goal is to explore how a pretrained vision-language model can produce primitive navigation behavior without explicit object detection or reinforcement learning.

---

## Key Characteristics

- **Robot:** Differential-drive robot created procedurally (no URDF dependency)  
- **Vision Language Model:** OpenAI CLIP (ViT-B/32) used as a perception backbone  
- **Action Expert:** Simple cosine-based navigation (sector-wise scoring + smooth motion)  
- **Environment:** PyBullet scene with random YCB objects, shapes, and textures  
- **Goal:** Reach the visual area corresponding to a given natural-language prompt  
---

## How It Works

At each simulation step:

1. The robot captures an RGB frame from its onboard camera.  
2. The image is split into *K* vertical sectors.  
3. Each sector is encoded using CLIP’s image encoder.  
4. The cosine similarity with the text embedding determines relevance.  
5. The sector with the highest score defines the heading.  
6. Forward velocity is scaled by similarity confidence.  
7. The robot moves smoothly toward the most semantically aligned region.

This setup forms a **minimal VLA loop**: language → vision → action.

---

## Architecture Schema

            +-----------------------------+
            |        Text Prompt          |
            |   e.g. "a red ball"         |
            +-------------+---------------+
                          |
                          v
            +-------------+---------------+
            |     CLIP Text Encoder       |
            |   (ViT-B/32 → embedding)    |
            +-------------+---------------+
                          |
                          v
    +----------------------------------------------+
    |              CLIP Navigation Loop            |
    | 1. Capture RGB frame                         |
    | 2. Split into K vertical sectors              |
    | 3. Encode each crop with CLIP Image Encoder   |
    | 4. Compare to text embedding (cosine score)   |
    | 5. Select argmax sector                      |
    | 6. Compute (v, ω) commands                   |
    +----------------------------------------------+
                          |
                          v
            +-------------+-------------+
            | PyBullet Differential Bot |
            +-------------+-------------+
                          |
                          v
            +-------------+-------------+
            |  Environment Update (GUI) |
            +---------------------------+


---

## Parameters

| Parameter | Description | Default |
|------------|-------------|----------|
| `K` | Number of image sectors | 9 |
| `fov_deg` | Camera field of view | 120° |
| `v_max` | Maximum forward velocity | 15.0 |
| `w_gain` | Angular velocity gain | 7.5 |
| `conf_th` | Confidence threshold | 0.08 |
| `SMOOTH_v` | Velocity smoothing factor | 0.15 |
| `SMOOTH_w` | Angular smoothing factor | 0.05 |

---

## Demo Videos

| File | Example Prompt | Description |
|------|----------------|-------------|
| demos/baseball.mp4 |  |  |
| demos/bleach.mp4 |  |  |
| demos/coffee.mp4 |  |  |
| demos/crackers.mp4 |  |  |
| demos/cyan_ball.mp4 |  |  |
| demos/fruits.mp4 |  |  |
| demos/lego.mp4 |  |  |
| demos/pear.mp4 |  |  |
| demos/plane.mp4 |  |  |
| demos/red_ball_demo.mp4 |  |  |
| demos/red_ball.mp4 |  |  |
| demos/red_ball2.mp4 |  |  |
| demos/sauce.mp4 |  |  |
| demos/searching_mug.mp4 |  |  |
| demos/strawberry.mp4 |  |  |
| demos/strawberry.mov |  |  |
| demos/sun.mov |  |  |
| demos/sun.mp4 |  |  |

---
## Installation

git clone https://github.com/rantaluca/MiniVLA-NAV.git
cd MiniVLA-NAV
pip install pybullet torch torchvision matplotlib pillow
pip install git+https://github.com/openai/CLIP.git

Python ≥ 3.9 is recommended.

---

## Running the System

### To test the agent:
python main.py

Then type a prompt when requested, the prompt should be the target object description: "a red ball on the ground"

### To visualize the environment:
python env.py –gui –n_objects 20



## Notes
- The project is intentionally lightweight: no RL training or external datasets.   
- The world generation can be deterministic when a seed is provided.  