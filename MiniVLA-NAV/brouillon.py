# ---------- Objects ----------
    def _spawn_random_objects(self, n: int):
        object_urdfs = [
            "duck_vhacd.urdf",
            "cloth.obj",
        ]

        for _ in range(n):
            # -------- sample a pose not too close to robot spawn --------
            for _try in range(100):
                x = self.rng.uniform(-self.arena + 0.2, self.arena - 0.2)
                y = self.rng.uniform(-self.arena + 0.2, self.arena - 0.2)
                if x * x + y * y > 0.6 ** 2:
                    break
            pos_xy = [x, y]
            yaw = self.rng.uniform(-math.pi, math.pi)
            orn = p.getQuaternionFromEuler([0, 0, yaw])

            # -------- choose primitive vs URDF --------
            if self.rng.random() > 0.5 and object_urdfs:
                # ---- URDF path ----
                urdf = self.rng.choice(object_urdfs)
                scale = self.rng.uniform(0.6, 1.4)
                # small lift so most assets don't intersect the plane
                base_pos = [pos_xy[0], pos_xy[1], 0.02]
                try:
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
                except Exception as e:
                    # fallback to a small box if an asset is missing on this install
                    hx, hy, hz = [self.rng.uniform(0.03, 0.08) for _ in range(3)]
                    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
                    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                                              rgbaColor=[0.6, 0.6, 0.6, 1])
                    z = max(0.02, hz)
                    bid = p.createMultiBody(
                        baseMass=0.5,
                        baseCollisionShapeIndex=col,
                        baseVisualShapeIndex=vis,
                        basePosition=[pos_xy[0], pos_xy[1], z],
                        baseOrientation=orn,
                    )
                    p.changeDynamics(bid, -1, lateralFriction=0.8, rollingFriction=0.1, spinningFriction=0.1)
                    self.object_ids.append(bid)
            else:
                # primitive choice
                shape = self.rng.choice(["box", "sphere", "cylinder"])
                color = [self.rng.uniform(0.1, 0.9) for _ in range(3)] + [1.0]

                if shape == "box":
                    hx, hy, hz = [self.rng.uniform(0.03, 0.12) for _ in range(3)]
                    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
                    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=color)
                    height = 2 * hz
                elif shape == "sphere":
                    r = self.rng.uniform(0.04, 0.12)
                    col = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
                    vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=color)
                    height = 2 * r
                else:  # cylinder
                    r = self.rng.uniform(0.04, 0.10)
                    h = self.rng.uniform(0.06, 0.20)
                    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
                    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=color)
                    height = h

                z = max(0.02, 0.5 * height)
                bid = p.createMultiBody(
                    baseMass=self.rng.choice([0.0, 0.5, 1.0]),  # some static, some dynamic
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=[pos_xy[0], pos_xy[1], z],
                    baseOrientation=orn,
                )
                p.changeDynamics(bid, -1, lateralFriction=0.8, rollingFriction=0.1, spinningFriction=0.1)
                self.object_ids.append(bid)


        image_folder = "textures/floors/" #folder in the local project directory

        # randomly select a texture file
        texture_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if texture_files:
            texture_file = os.path.join(image_folder, self.rng.choice(texture_files))
            texture_id = p.loadTexture(texture_file)
            p.changeVisualShape(0, -1, textureUniqueId=texture_id)