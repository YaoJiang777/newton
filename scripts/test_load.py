# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Sim Cloth Bending
#
# This simulation demonstrates cloth bending behavior using the Vertex Block
# Descent (VBD) integrator. A cloth mesh, initially curved, is dropped on
# the ground. The cloth maintains its curved shape due to bending stiffness,
# controlled by edge_ke and edge_kd parameters.
#
###########################################################################

import time
import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples

# from .xpbd_modified import SolverXPBD
# from newton.solvers import SolverXPBD
from .xpbd_modified2 import SolverXPBD


class Example:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        self.dt = self.frame_dt


        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = 10

        self.viewer = viewer


        builder = newton.ModelBuilder()

        path = "/mnt/d/Columbia/6998 Graphics and computational Motions/Final Project/Git codes/newton/assets/test.usda"

        builder.add_usd(
            source=path,
            only_load_enabled_joints=False,
        )

        builder.add_ground_plane()
        self.model = builder.finalize()


        from .rod_constraints import parse_rod_constraint, update_joint

        rod_constraint = parse_rod_constraint(source=path)
        update_joint(self.model, rod_constraint)

        # Sanity Check Print
        # print("particle_q:")
        # print(self.model.particle_q)
        
        # print("body_shapes:")
        # print(self.model.body_shapes)

        # print("body_key:")
        # print(self.model.body_key)
        
        # print("joint_key:")
        # print(self.model.joint_key)

        # print("joint_type:")
        # print(self.model.joint_type)

        # props = [
        #     "joint_key",
        #     "joint_type",
        #     "joint_enabled",
        #     "joint_parent",
        #     "joint_child",
        #     "joint_X_p",
        #     "joint_X_c",
        #     "joint_limit_lower",
        #     "joint_limit_upper",
        #     "joint_qd_start",
        #     "joint_dof_dim",
        #     "joint_dof_mode",
        #     "joint_axis",
        #     "joint_target_ke",
        #     "joint_target_kd",
        #     "rod_length"

        # ]

        # for prop in props:
        #     val = getattr(self.model, prop)
        #     print(prop, "=", val)
        #     if prop == "joint_axis":
        #         print(np.shape(val))

        
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 1.0e0
        self.model.soft_contact_mu = 1.0

        self.solver = SolverXPBD(self.model)
        

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)
        self.capture()

    def test(self):
        state_in = self.state_0
        state_out = self.state_1

        for i in range(100):
            self.solver.step(state_in, state_out, self.control, self.contacts, self.dt)
            state_in, state_out = state_out, state_in

        # raise ValueError
            # print("i:", i)
            # print("state_in:", state_in)
            # print("state_out:", state_out)
        
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 20,
            show_body_q = True,
            show_body_qd = True,
        )

    
    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    # def test(self):
    #     newton.examples.test_particle_state(
    #         self.state_0,
    #         "particles have come close to a rest",
    #         lambda q, qd: max(abs(qd)) < 0.1,
    #     )

    #     p_lower = wp.vec3(-3.0, -3.0, 0.0)
    #     p_upper = wp.vec3(3.0, 3.0, 2.0)
    #     newton.examples.test_particle_state(
    #         self.state_0,
    #         "particles are within a reasonable volume",
    #         lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
    #     )

    #     newton.examples.test_particle_state(
    #         self.state_0,
    #         "lower particles touch the ground",
    #         lambda q, qd: q[2] < 0.15,
    #         indices=[4, 5, 12, 13],
    #     )


if __name__ == "__main__":

    viewer = newton.viewer.ViewerGL(headless=False)

    example = Example(viewer)
    # example.test()

    while example.viewer.is_running():
        if not example.viewer.is_paused():
            with wp.ScopedTimer("step", active=False):
                example.step()

        with wp.ScopedTimer("render", active=False):
            example.render()

        time.sleep(0.2)

