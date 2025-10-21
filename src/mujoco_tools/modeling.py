import logging
import mujoco
red   = [1,0,0, 0.6]
green = [0,1,0, 0.6]
blue  = [0,0,1, 0.6]
gray  = [0.8, 0.8, 0.8, 0.6]

def sample_manipulator(spec, *, prefix=None, free_joint=False, contype=1, conaffinity=1):
    if prefix is None:
        prefix = ""
    # base_link
    base = spec.worldbody.add_body(name=f"{prefix}base_link", pos=[0, 0, 0])
    if free_joint:
        free_joint = base.add_freejoint() # add a free joint
        free_joint.name = f"{prefix}root"
    base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.02], rgba=gray, contype=contype, conaffinity=conaffinity)

    # link0
    link0 = base.add_body(name=f"{prefix}link0", mass=0.5, inertia=[8e-4, 8e-4, 8e-4], pos=[0, 0, 0.02])
    link0.add_joint(name=f"{prefix}joint0", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], range=[-90, 90]) # joint range unit is [degree]
    # link1
    link1 = link0.add_body(name=f"{prefix}link1", mass=0.3, inertia=[4e-4, 4e-4, 4e-4])
    link1.add_joint(name=f"{prefix}joint1", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[-90, 90])
    # link2
    link2 = link1.add_body(name=f"{prefix}link2", mass=0.2, inertia=[2e-4, 2e-4, 2e-4])
    link2.add_joint(name=f"{prefix}joint2", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-90, 90])
    link2.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[0,0,0, 0,0,0.25], size=[0.02, 0.0, 0.0], rgba=red, contype=contype, conaffinity=conaffinity)
    # link3
    link3 = link2.add_body(name=f"{prefix}link3", mass=0.2, inertia=[2e-4, 2e-4, 2e-4], pos = [0, 0, 0.3])
    link3.add_joint(name=f"{prefix}joint3", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-90, 90])
    link3.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[0,0,0, 0,0,0.25], size=[0.02, 0.0, 0.0], rgba=green, contype=contype, conaffinity=conaffinity)
    # link4
    link4 = link3.add_body(name=f"{prefix}link4", mass=0.1, inertia=[1e-4, 1e-4, 1e-4], pos=[0, 0, 0.3])
    link4.add_joint(name=f"{prefix}joint4", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], range=[-90, 90])
    # link5
    link5 = link4.add_body(name=f"{prefix}link5", mass=0.1, inertia=[1e-4, 1e-4, 1e-4])
    link5.add_joint(name=f"{prefix}joint5", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-90, 90])
    # link6
    link6 = link5.add_body(name=f"{prefix}link6", mass=0.1, inertia=[1e-4, 1e-4, 1e-4])
    link6.add_joint(name=f"{prefix}joint6", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[-90, 90])
    link6.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[0,0,0, 0,0,0.1], size=[0.02, 0.0, 0.0], rgba=blue, contype=contype, conaffinity=conaffinity)
    # end effector
    link6.add_site(name=f"{prefix}ee", pos=[0, 0, 0.12], size=[0.01, 0.0, 0.0])

def sample_robot_biped(spec, *, prefix=None, free_joint=True, contype=1, conaffinity=1):
    if prefix is None:
        prefix = ""
    # base_link
    base = spec.worldbody.add_body(name=f"{prefix}base_link", pos=[0, 0, 0.05])
    if free_joint:
        free_joint = base.add_freejoint() # add a free joint
        free_joint.name = f"{prefix}root"
    base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.02], rgba=gray, contype=contype, conaffinity=conaffinity)

    # rleg_link0
    rleg_link0 = base.add_body(name=f"{prefix}rleg_link0", mass=0.5, inertia=[8e-4, 8e-4, 8e-4], pos=[0, -0.1, 0])
    rleg_link0.add_joint(name=f"{prefix}rleg_joint0", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], range=[-90, 90])
    # rleg_link1
    rleg_link1 = rleg_link0.add_body(name=f"{prefix}rleg_link1", mass=0.3, inertia=[4e-4, 4e-4, 4e-4])
    rleg_link1.add_joint(name=f"{prefix}rleg_joint1", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[-90, 90])
    # rleg_link2
    rleg_link2 = rleg_link1.add_body(name=f"{prefix}rleg_link2", mass=0.2, inertia=[2e-4, 2e-4, 2e-4])
    rleg_link2.add_joint(name=f"{prefix}rleg_joint2", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-90, 90])
    rleg_link2.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[0,0,0, 0,0,-0.20], size=[0.02, 0.0, 0.0], rgba=red, contype=contype, conaffinity=conaffinity)
    # rleg_link3
    rleg_link3 = rleg_link2.add_body(name=f"{prefix}rleg_link3", mass=0.2, inertia=[2e-4, 2e-4, 2e-4], pos = [0, 0, -0.25])
    rleg_link3.add_joint(name=f"{prefix}rleg_joint3", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-90, 90])
    rleg_link3.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[0,0,0, 0,0,-0.20], size=[0.02, 0.0, 0.0], rgba=green, contype=contype, conaffinity=conaffinity)
    # rleg_link4
    rleg_link4 = rleg_link3.add_body(name=f"{prefix}rleg_link4", mass=0.1, inertia=[1e-4, 1e-4, 1e-4], pos=[0, 0, -0.25])
    rleg_link4.add_joint(name=f"{prefix}rleg_joint4", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-90, 90])
    # rleg_link5
    rleg_link5 = rleg_link4.add_body(name=f"{prefix}rleg_link5", mass=0.1, inertia=[1e-4, 1e-4, 1e-4])
    rleg_link5.add_joint(name=f"{prefix}rleg_joint5", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[-90, 90])
    rleg_link5.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[-0.05,0,0, 0.1,0,0], size=[0.02, 0.0, 0.0], rgba=blue)
    # right end effector
    rleg_link5.add_site(name=f"{prefix}right_ee", pos=[0.12, 0, 0], size=[0.01, 0.0, 0.0])

    # lleg_link0
    lleg_link0 = base.add_body(name=f"{prefix}lleg_link0", mass=0.5, inertia=[8e-4, 8e-4, 8e-4], pos=[0, 0.1, 0])
    lleg_link0.add_joint(name=f"{prefix}lleg_joint0", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], range=[-90, 90])
    # lleg_link1
    lleg_link1 = lleg_link0.add_body(name=f"{prefix}lleg_link1", mass=0.3, inertia=[4e-4, 4e-4, 4e-4])
    lleg_link1.add_joint(name=f"{prefix}lleg_joint1", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[-90, 90])
    # lleg_link2
    lleg_link2 = lleg_link1.add_body(name=f"{prefix}lleg_link2", mass=0.2, inertia=[2e-4, 2e-4, 2e-4])
    lleg_link2.add_joint(name=f"{prefix}lleg_joint2", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-90, 90])
    lleg_link2.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[0,0,0, 0,0,-0.20], size=[0.02, 0.0, 0.0], rgba=red, contype=contype, conaffinity=conaffinity)
    # lleg_link3
    lleg_link3 = lleg_link2.add_body(name=f"{prefix}lleg_link3", mass=0.2, inertia=[2e-4, 2e-4, 2e-4], pos = [0, 0, -0.25])
    lleg_link3.add_joint(name=f"{prefix}lleg_joint3", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-90, 90])
    lleg_link3.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[0,0,0, 0,0,-0.20], size=[0.02, 0.0, 0.0], rgba=green, contype=contype, conaffinity=conaffinity)
    # lleg_link4
    lleg_link4 = lleg_link3.add_body(name=f"{prefix}lleg_link4", mass=0.1, inertia=[1e-4, 1e-4, 1e-4], pos=[0, 0, -0.25])
    lleg_link4.add_joint(name=f"{prefix}lleg_joint4", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=[-90, 90])
    # lleg_link5
    lleg_link5 = lleg_link4.add_body(name=f"{prefix}lleg_link5", mass=0.1, inertia=[1e-4, 1e-4, 1e-4])
    lleg_link5.add_joint(name=f"{prefix}lleg_joint5", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], range=[-90, 90])
    lleg_link5.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, fromto=[-0.05,0,0, 0.1,0,0], size=[0.02, 0.0, 0.0], rgba=blue, contype=contype, conaffinity=conaffinity)
    # left end effector
    lleg_link5.add_site(name=f"{prefix}left_ee", pos=[0.12, 0, 0], size=[0.01, 0.0, 0.0])