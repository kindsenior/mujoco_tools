#!/usr/bin/env -S python3 -i
import logging
# logging.basicConfig(level=logging.DEBUG)
import numpy as np
import mujoco
import mujoco.viewer

red   = [1,0,0, 0.6]
green = [0,1,0, 0.6]
blue  = [0,0,1, 0.6]
gray  = [0.8, 0.8, 0.8, 0.6]

def sample_manipulator():
    spec = mujoco.MjSpec()

    # base_link
    base = spec.worldbody.add_body(name="base_link", pos=[0, 0, 0])
    base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.02], rgba=gray)

    # link0
    link0 = base.add_body(name="link0", mass=0.5, inertia=[8e-4, 8e-4, 8e-4], pos=[0, 0, 0.02])
    link0.add_joint(name="joint0", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], range=[-np.pi, np.pi])
    # link1
    link1 = link0.add_body(name="link1", mass=0.3, inertia=[4e-4, 4e-4, 4e-4])
    link1.add_joint(name="joint1", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[-np.pi, np.pi])
    # link2
    link2 = link1.add_body(name="link2", mass=0.2, inertia=[2e-4, 2e-4, 2e-4])
    link2.add_joint(name="joint2", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-np.pi, np.pi])
    link2.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[0, 0, 0, 0, 0, 0.3], size=[0.02, 0.0, 0.0], rgba=red)
    # link3
    link3 = link2.add_body(name="link3", mass=0.2, inertia=[2e-4, 2e-4, 2e-4], pos = [0, 0, 0.3])
    link3.add_joint(name="joint3", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-np.pi, np.pi])
    link3.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[0, 0, 0, 0, 0, 0.3], size=[0.02, 0.0, 0.0], rgba=green)
    # link4
    link4 = link3.add_body(name="link4", mass=0.1, inertia=[1e-4, 1e-4, 1e-4], pos=[0, 0, 0.3])
    link4.add_joint(name="joint4", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], range=[-np.pi, np.pi])
    # link5
    link5 = link4.add_body(name="link5", mass=0.1, inertia=[1e-4, 1e-4, 1e-4])
    link5.add_joint(name="joint5", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-np.pi, np.pi])
    # link6
    link6 = link5.add_body(name="link6", mass=0.1, inertia=[1e-4, 1e-4, 1e-4])
    link6.add_joint(name="joint6", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[-np.pi, np.pi])
    link6.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[0,0,0, 0,0,0.1], size=[0.02, 0.0, 0.0], rgba=blue)
    # end effector
    link6.add_site(name="ee", pos=[0, 0, 0.12], size=[0.01, 0.0, 0.0])

    # compile model
    model = spec.compile()
    data = mujoco.MjData(model)

    return model, spec, data

def sample_robot_biped():
    spec = mujoco.MjSpec()

    # base_link
    base = spec.worldbody.add_body(name="base_link", pos=[0, 0, 0.05])
    free_joint = base.add_freejoint() # add a free joint
    free_joint.name = "root"
    base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.02], rgba=gray)

    # rleg_link0
    rleg_link0 = base.add_body(name="rleg_link0", mass=0.5, inertia=[8e-4, 8e-4, 8e-4], pos=[0, -0.1, 0])
    rleg_link0.add_joint(name="rleg_joint0", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], range=[-np.pi, np.pi])
    # rleg_link1
    rleg_link1 = rleg_link0.add_body(name="rleg_link1", mass=0.3, inertia=[4e-4, 4e-4, 4e-4])
    rleg_link1.add_joint(name="rleg_joint1", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[-np.pi, np.pi])
    # rleg_link2
    rleg_link2 = rleg_link1.add_body(name="rleg_link2", mass=0.2, inertia=[2e-4, 2e-4, 2e-4])
    rleg_link2.add_joint(name="rleg_joint2", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-np.pi, np.pi])
    rleg_link2.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[0, 0, 0, 0, 0, -0.25], size=[0.02, 0.0, 0.0], rgba=red)
    # rleg_link3
    rleg_link3 = rleg_link2.add_body(name="rleg_link3", mass=0.2, inertia=[2e-4, 2e-4, 2e-4], pos = [0, 0, -0.25])
    rleg_link3.add_joint(name="rleg_joint3", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-np.pi, np.pi])
    rleg_link3.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[0, 0, 0, 0, 0, -0.25], size=[0.02, 0.0, 0.0], rgba=green)
    # rleg_link4
    rleg_link4 = rleg_link3.add_body(name="rleg_link4", mass=0.1, inertia=[1e-4, 1e-4, 1e-4], pos=[0, 0, -0.25])
    rleg_link4.add_joint(name="rleg_joint4", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-np.pi, np.pi])
    # rleg_link5
    rleg_link5 = rleg_link4.add_body(name="rleg_link5", mass=0.1, inertia=[1e-4, 1e-4, 1e-4])
    rleg_link5.add_joint(name="rleg_joint5", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-np.pi, np.pi])
    rleg_link5.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[0.1, 0, 0, 0, 0, 0], size=[0.02, 0.0, 0.0], rgba=blue)
    # right end effector
    rleg_link5.add_site(name="right_ee", pos=[0.12, 0, 0], size=[0.01, 0.0, 0.0])

    # lleg_link0
    lleg_link0 = base.add_body(name="lleg_link0", mass=0.5, inertia=[8e-4, 8e-4, 8e-4], pos=[0, 0.1, 0])
    lleg_link0.add_joint(name="lleg_joint0", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], range=[-np.pi, np.pi])
    # lleg_link1
    lleg_link1 = lleg_link0.add_body(name="lleg_link1", mass=0.3, inertia=[4e-4, 4e-4, 4e-4])
    lleg_link1.add_joint(name="lleg_joint1", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[-np.pi, np.pi])
    # lleg_link2
    lleg_link2 = lleg_link1.add_body(name="lleg_link2", mass=0.2, inertia=[2e-4, 2e-4, 2e-4])
    lleg_link2.add_joint(name="lleg_joint2", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-np.pi, np.pi])
    lleg_link2.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[0, 0, 0, 0, 0, -0.25], size=[0.02, 0.0, 0.0], rgba=red)
    # lleg_link3
    lleg_link3 = lleg_link2.add_body(name="lleg_link3", mass=0.2, inertia=[2e-4, 2e-4, 2e-4], pos = [0, 0, -0.25])
    lleg_link3.add_joint(name="lleg_joint3", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-np.pi, np.pi])
    lleg_link3.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[0, 0, 0, 0, 0, -0.25], size=[0.02, 0.0, 0.0], rgba=green)
    # lleg_link4
    lleg_link4 = lleg_link3.add_body(name="lleg_link4", mass=0.1, inertia=[1e-4, 1e-4, 1e-4], pos=[0, 0, -0.25])
    lleg_link4.add_joint(name="lleg_joint4", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-np.pi, np.pi])
    # lleg_link5
    lleg_link5 = lleg_link4.add_body(name="lleg_link5", mass=0.1, inertia=[1e-4, 1e-4, 1e-4])
    lleg_link5.add_joint(name="lleg_joint5", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-np.pi, np.pi])
    lleg_link5.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[0.1, 0, 0, 0, 0, 0], size=[0.02, 0.0, 0.0], rgba=blue)
    # left end effector
    lleg_link5.add_site(name="left_ee", pos=[0.12, 0, 0], size=[0.01, 0.0, 0.0])

    # compile model
    model = spec.compile()
    data = mujoco.MjData(model)

    return model, spec, data

def test():
    # create model
    global model, data
    model, spec, data = sample_robot_biped()

    # FK
    data.qpos[7:] = np.deg2rad([0,0,-30, 60, -30,0, 0,0,-30, 60, -30,0])
    mujoco.mj_forward(model, data)
    for ee_name in ["right_ee", "left_ee"]:
        print(f"{ee_name} pos (world):", data.site(ee_name).xpos.copy())

    # export to xml
    xml_str = spec.to_xml()
    with open("sample_robot_biped.xml", "w", encoding="utf-8") as f:
        f.write(xml_str)

    # display
    global viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1 # visualize joints
    viewer.sync()

if __name__ == "__main__":
    test()