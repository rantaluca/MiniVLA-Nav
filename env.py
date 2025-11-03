#!/usr/bin/env python3
"""
env.py

Spawn a minimal differential-drive robot in a randomized 3D scene (PyBullet).
- Lightweight: programmatically built with createMultiBody (no URDF needed).
- Simple velocity interface: set_cmd(v, omega) in SI units.
- Random objects: boxes/spheres/cylinders with random sizes/poses/colors.
- Optional keyboard teleop (z/s: linear, q/d: turn, x: stop).

Usage:
python env.py --gui            # with viewer
python env.py --n_objects 30   # more clutter
python env.py --seed 42

This environment is inspired by 
"""

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

# display camera feed in GUI mode
import cv2
import threading
import matplotlib.pyplot as plt


@dataclass
class RobotParams:
    def __init__(self):
        # default parameters
        self.base_size: Tuple[float, float, float] = (0.30, 0.22, 0.10)  # L, W, H (m)
        self.wheel_radius: float = 0.075                                  # m
        self.wheel_thickness: float = 0.028                              # m
        self.track_width: float = 0.24                                   # distance between wheel centers (m)
        self.base_mass: float = 2.0                                      # kg
        self.wheel_mass: float = 0.15                                    # kg
        self.max_wheel_speed: float = 20.0                               # rad/s (for teleop clamp)
        self.wheel_kp: float = 1.0                                       # motor velocity P gain


class DiffBotEnv:
    def __init__(
        self,
        gui: bool = True,
        seed: int = None,
        n_objects: int = 10,
        arena_half_size: float = 4.5,
        robot_params: RobotParams = RobotParams(),
        gravity: float = -9.81,
        time_step: float = 1.0 / 240.0,
    ):
        self.gui = gui
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.N = n_objects
        self.arena = float(arena_half_size)
        self.params = robot_params
        self.dt = time_step

        self._connect()
        self._setup_world(gravity)
        self._spawn_plane()
        self.robot_id, (self.left_joint, self.right_joint) = self._spawn_diffbot()
        self.object_ids: List[int] = []
        self._spawn_random_objects(self.N)

        # internal state for convenience
        self.last_cmd = (0.0, 0.0)  # (v, omega)
        self.left_w = 0.0
        self.right_w = 0.0

    # World setup and utilities
    def _connect(self):
        if self.gui:
            self.cid = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _setup_world(self, gravity: float):
        p.resetSimulation()
        p.setGravity(0, 0, gravity)
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt, numSolverIterations=150)

        if self.gui:
            # camera: slightly elevated, looking toward origin
            p.resetDebugVisualizerCamera(
                cameraDistance=4.0,
                cameraYaw=90,
                cameraPitch=-40,
                cameraTargetPosition=[0, 0, 0],
            )

    def step(self, steps: int = 1, cmd: Tuple[float, float] = None, camera_feed: bool = False):
        if cmd is not None:
            self.set_cmd(cmd[0], cmd[1])
        for _ in range(steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(self.dt)
        if camera_feed:
            return self._get_camera_image()
        
    
    # Scene generation utilities
    def _spawn_plane(self):
        p.loadURDF("plane.urdf")
        # randomly select a texture file
        floor_image_folder = "textures/floors/" 
        texture_files = [f for f in os.listdir(floor_image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if texture_files:
            # random floor texture
            ground_texture_file = os.path.join(floor_image_folder, self.rng.choice(texture_files))
            ground_texture_id = p.loadTexture(ground_texture_file)
            # lower the brightness a bit
            p.changeVisualShape(0, -1, textureUniqueId=ground_texture_id, rgbaColor=[0.7, 0.7, 0.7, 1.0])

        # # spawn walls
        # height = 3.0
        # L = self.arena
        # pts = [
        #     [-L, -L, height], [L, -L, height],
        #     [L, -L, height], [L, L, height],
        #     [L, L, height], [-L, L, height],
        #     [-L, L, height], [-L, -L, height],
        # ]

        # wall_image_folder = "textures/walls/"
        # wall_texture_files = [f for f in os.listdir(wall_image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        # if wall_texture_files:
        #     wall_texture_file = os.path.join(wall_image_folder, self.rng.choice(wall_texture_files))
        #     wall_texture_id = p.loadTexture(wall_texture_file)
        
        # for i in range(0, len(pts), 2):
        #     col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L, 0.1, height / 2])
        #     vis = -1
        #     # random wall texture
        #     if wall_texture_files:
        #         vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[L, 0.1, height / 2])
        #     pos = [(pts[i][0] + pts[i+1][0]) / 2, (pts[i][1] + pts[i+1][1]) / 2, height / 2]
        #     orn = p.getQuaternionFromEuler([0, 0, math.atan2(pts[i+1][1] - pts[i][1], pts[i+1][0] - pts[i][0])])
        #     wall_id = p.createMultiBody(
        #         baseMass=0.0,
        #         baseCollisionShapeIndex=col,
        #         baseVisualShapeIndex=vis,
        #         basePosition=pos,
        #         baseOrientation=orn,
        #     )

        #     if vis != -1 and wall_texture_files:
        #         # fill the wall with texture trying to avoid stretching
        #         p.changeVisualShape(wall_id, -1, textureUniqueId=wall_texture_id, rgbaColor=[0.7, 0.7, 0.7, 1.0])

    def _spawn_random_objects(self, n: int):
            # object_urdfs = [
            #     "duck_vhacd.urdf",
            #     "objects/mug.urdf",
            #     "lego/lego.urdf",
            #     "soccerball.urdf",  
            #     "urdfs/dinnerware/pan_tefal.urdf",
            #     "urdfs/dinnerware/plate.urdf",
            # ]

            # ajouter tout les urdf dans le dossier urdfs/ycb_assets/
            ycb_urdfs = [os.path.join("urdfs/ycb/ycb_assets/", f) for f in os.listdir("urdfs/ycb/ycb_assets/") if f.endswith('.urdf')]
            #print("YCB URDFS:", ycb_urdfs)
            object_urdfs = ycb_urdfs

            for _ in range(n):
                # finding a valid position
                for _try in range(100):
                    x = self.rng.uniform(-self.arena + 0.2, self.arena - 0.2)
                    y = self.rng.uniform(-self.arena + 0.2, self.arena - 0.2)
                    if x * x + y * y > 0.6 ** 2:
                        break
                pos_xy = [x, y]
                yaw = self.rng.uniform(-math.pi, math.pi)
                orn = p.getQuaternionFromEuler([0, 0, yaw])

                #Choose between URDF object or primitive shape
                if self.rng.random() > 0.99 and object_urdfs:
                    
                    # urdf object
                    urdf = self.rng.choice(object_urdfs)
                    scale = self.rng.uniform(0.2, 0.7)
                    base_pos = [pos_xy[0], pos_xy[1], 0.2]
                    
                    bid = p.loadURDF(
                        urdf,
                        base_pos,
                        orn,
                        useFixedBase=False,
                        globalScaling=scale,
                    )
                    
                    # friction tweak
                    p.changeDynamics(bid, -1, lateralFriction=0.8, rollingFriction=0.1, spinningFriction=0.1)
                    self.object_ids.append(bid)

                else:
                    # primitive choice
                    shape = self.rng.choice(["box", "sphere", "cylinder"])
                    color = [self.rng.uniform(0.1, 0.9) for _ in range(3)] + [1.0]

                    if shape == "box":
                        hx, hy, hz = [self.rng.uniform(0.05, 0.50) for _ in range(3)]
                        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
                        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=color)
                        height = 2 * hz
                    elif shape == "sphere":
                        r = self.rng.uniform(0.05, 0.50)
                        col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
                        vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=color)
                        height = 2 * r
                    else:  # cylinder
                        r = self.rng.uniform(0.05, 0.50)
                        h = self.rng.uniform(0.05, 0.50)
                        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
                        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=color)
                        height = h

                    z = max(0.02, 0.5 * height)
                    bid = p.createMultiBody(
                        baseMass=self.rng.choice([0.0, 0.5, 1.0]),  
                        baseCollisionShapeIndex=col,
                        baseVisualShapeIndex=vis,
                        basePosition=[pos_xy[0], pos_xy[1], z],
                        baseOrientation=orn,
                    )
                    p.changeDynamics(bid, -1, lateralFriction=0.8, rollingFriction=0.1, spinningFriction=0.1)
                    self.object_ids.append(bid)

    def _spawn_diffbot(self):
        """
        Create a minimal diff-drive robot: base + 2 revolute wheels + 2 spherical caster.
        Uses createMultiBody directly (no URDF).
        """
        P = self.params
        L, W, H = P.base_size

        # geometry parames 
        rear_x_offset = 0.0   
        caster_radius = 0.03         
        caster_x = +0.12         

        # base
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L/2, W/2, H/2])
        base_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[L/2, W/2, H/2], rgbaColor=[0.45, 0.45, 0.4, 1]
        )

        # Position base 
        base_pos = [1.0, 0, P.wheel_radius + 0.5 * H]
        base_orn = p.getQuaternionFromEuler([0, 0, math.pi])

        # Wheels
        wheel_quat = p.getQuaternionFromEuler([-math.pi / 2, 0, 0])
        wheel_col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=P.wheel_radius, height=P.wheel_thickness,
            collisionFrameOrientation=wheel_quat
        )
        wheel_vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=P.wheel_radius, length=P.wheel_thickness,
            visualFrameOrientation=wheel_quat, rgbaColor=[0.1, 0.1, 0.1, 1]
        )

        # Caster ball (sphere)
        caster_col = p.createCollisionShape(p.GEOM_SPHERE, radius=caster_radius)
        caster_vis = p.createVisualShape(p.GEOM_SPHERE, radius=caster_radius, rgbaColor=[0.15, 0.15, 0.18, 1])

        # Link arrays
        link_masses = [P.wheel_mass, P.wheel_mass, 0.02, 0.02]  # light caster
        link_cols   = [wheel_col,   wheel_col,   caster_col, caster_col]
        link_viss   = [wheel_vis,   wheel_vis,   caster_vis, caster_vis]

        # Place wheels a bit back (rear_x_offset), left/right on ±track/2, caster in front center
        link_pos = [
            [rear_x_offset, +P.track_width/2.0, 0.0],   # left wheel
            [rear_x_offset, -P.track_width/2.0, 0.0],   # right wheel
            [caster_x, 0.0, -P.wheel_radius/2-0.006],  # caster1
            [-caster_x, 0.0, -P.wheel_radius/2-0.006],  # caster2
        ]
        link_orns = [[0,0,0,1]] * 4

        # Inertial frames (basic setup)
        link_inert_pos = [[0,0,0]] * 4
        link_inert_orn = [[0,0,0,1]] * 4

        # all 4 are direct children of base (0)
        link_parent     = [0, 0, 0, 0]

        # Joints:
        # wheels: REVOLUTE around Y
        # caster: SPHERICAL (free to swivel)
        link_joint_types = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_SPHERICAL, p.JOINT_SPHERICAL]
        link_joint_axes  = [[0,1,0],[0,1,0],[0,0,0],[0,0,0]]  # axis ignored for spherical

        robot_id = p.createMultiBody(
            baseMass=P.base_mass,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=base_pos,
            baseOrientation=base_orn,
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_cols,
            linkVisualShapeIndices=link_viss,
            linkPositions=link_pos,
            linkOrientations=link_orns,
            linkInertialFramePositions=link_inert_pos,
            linkInertialFrameOrientations=link_inert_orn,
            linkParentIndices=link_parent,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes,
        )

        # Enable motors 
        for j in [0, 1]:
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=2.0,
                positionGain=0.0,
                velocityGain=self.params.wheel_kp,
            )

        # More friction on caster
        try:
            for j in [2, 3]:
                p.changeDynamics(robot_id, j, lateralFriction=0.8, rollingFriction=0.1, spinningFriction=0.1)
        except Exception:
            pass

        # Camera position
        self._cam_rel_pos = [0.10, 0.0, 0.08]      # forward & slightly above base center
        self._cam_rel_rpy = [-0.15, 0.0, 0.0]      # slight pitch down toward the floor

        # Return robot and the indices of left/right wheel joints
        return robot_id, (0, 1)

    # Control interface
    def set_cmd(self, v: float, omega: float):
        """
        Command linear velocity v [m/s] and angular velocity omega [rad/s].
        """
        P = self.params
        r = P.wheel_radius
        L = P.track_width

        wl = (v - 0.5 * omega * L) / r
        wr = (v + 0.5 * omega * L) / r

        # Clamp to motor limits
        wl = float(np.clip(wl, -P.max_wheel_speed, P.max_wheel_speed))
        wr = float(np.clip(wr, -P.max_wheel_speed, P.max_wheel_speed))

        self.left_w, self.right_w = wl, wr
        self.last_cmd = (v, omega)

        # Apply motor velocity control
        p.setJointMotorControl2(self.robot_id, self.left_joint,  p.VELOCITY_CONTROL, targetVelocity=wl, force=4.0)
        p.setJointMotorControl2(self.robot_id, self.right_joint, p.VELOCITY_CONTROL, targetVelocity=wr, force=4.0)

    # State utilities
    def _get_camera_image(self):
        # robot pose
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)

        # CAM POSE
        cam_local = [0.25, 0.0, 0.20]       
        look_local = [0.90, 0.0, 0.15]      

        # world coordinates
        cam_world, _  = p.multiplyTransforms(pos, orn, cam_local,  [0, 0, 0, 1])
        look_world, _ = p.multiplyTransforms(pos, orn, look_local, [0, 0, 0, 1])

        # view & projection
        view = p.computeViewMatrix(cam_world, look_world, [0, 0, 1])

        width, height = 720, 480
        proj = p.computeProjectionMatrixFOV(
            fov=70, aspect=width/height, nearVal=0.02, farVal=20.0
        )

        # render 
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        w, h, rgba, _, _ = p.getCameraImage(width, height, view, proj, renderer=renderer)

        # convert to OpenCV BGR
        frame = np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)
        bgr = frame[:, :, :3][:, :, ::-1].copy()
        return bgr
    
    def get_pose(self) -> Tuple[float, float, float]:
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return float(pos[0]), float(pos[1]), float(yaw)

    def reset_pose(self, x=0.0, y=0.0, yaw=0.0):
        z = self.params.wheel_radius + 0.5 * self.params.base_size[2]
        q = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robot_id, [x, y, z], q)

    # Teleop utility
    def teleop_once(self, v_step=0.08, w_step=0.6):
        """
        ESDF keys to drive, X to stop, R to reset
        """
        events = p.getKeyboardEvents()
        v, w = self.last_cmd
        for k, s in events.items():
            if s & p.KEY_WAS_TRIGGERED or s & p.KEY_IS_DOWN:
                if k in (ord('e'), ord('E')):
                    v += v_step
                elif k in (ord('d'), ord('D')):
                    v -= v_step
                elif k in (ord('s'), ord('S')):
                    w += w_step
                elif k in (ord('f'), ord('F')):
                    w -= w_step
                elif k in (ord('x'), ord('X')):
                    v, w = 0.0, 0.0
                elif k in (ord('r'), ord('R')):
                    self.reset_pose(0.0, 0.0, 0.0)

        self.set_cmd(v, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true", help="Run with PyBullet GUI")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n_objects", type=int, default=20)
    ap.add_argument("--arena", type=float, default=4.5, help="Arena size (m)")
    args = ap.parse_args()

    scene = DiffBotEnv(
        gui=args.gui,
        seed=args.seed,
        n_objects=args.n_objects,
        arena_half_size=args.arena,
    )

    print("DiffBot scene ready ! ")
    # print("API:")
    # print("  - scene.set_cmd(v, omega)  # m/s, rad/s")
    # print("  - scene.get_pose()         # (x, y, yaw)")
    print("E forward/D backward, S left/Fx right, X stop, R reset, ESC quit")

    if args.gui:
        scene.set_cmd(0.0, 0.0)

        # Create a resizable window once (prevents Mac freeze issues)
        cv2.namedWindow("DiffBot Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DiffBot Camera", 448, 448)

        while True:
            scene.teleop_once()
            frame_bgr = scene.step(1, camera_feed=True)  # returns BGR

            if frame_bgr is not None:
                cv2.imshow("DiffBot Camera", frame_bgr)

            # Exit on ESC or when the window is closed
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or cv2.getWindowProperty("DiffBot Camera", cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()