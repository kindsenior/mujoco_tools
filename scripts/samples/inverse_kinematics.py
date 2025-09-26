#!/usr/bin/env -S python3 -i
import logging
import numpy as np
import mujoco
import mujoco.viewer
from mujoco_tools.kinematics import *

def test_ik(*, pos=[0.3,0.3,0.3], rpy=[0,0,np.pi/4], target_name="ee", goal_name="ik_goal", **kwargs):
    # set target
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, goal_name)
    mid = model.body_mocapid[bid] # convert body id to mocap id (mocaps also belong to body class)
    data.mocap_pos[mid] = pos
    mujoco.mju_euler2Quat(data.mocap_quat[mid], rpy, "xyz")
    logging.info("goal pos: %s", data.mocap_pos[mid])
    logging.info("goal quat: %s", data.mocap_quat[mid])

    # IK
    viewer.user_scn.ngeom = 0 # reset the number of geoms
    ok, iters = inverse_kinematics(model, data, site_name=target_name, goal_name=goal_name,
                                   max_iters=200, tol_pos=1e-6, damping=1e-3, step_size=0.7,
                                   viewer=viewer, **kwargs)
    print("IK success:", ok, "iters:", iters)

    viewer.sync()

if __name__ == "__main__":
    # setup logging
    logger = logging.getLogger("mujoco_tools.kinematics")
    logger.setLevel(logging.INFO)

    # create model
    from mujoco_tools.modeling import sample_manipulator
    spec = mujoco.MjSpec()
    sample_manipulator(spec, free_joint=True) # add a free joint to the base link

    # add the goal mocap
    goal_body = spec.worldbody.add_body(name="ik_goal", mocap=True) # set to mocap body

    # build
    global model, data
    model = spec.compile()
    data = mujoco.MjData(model)

    global q_mask, v_mask
    q_mask, v_mask, _, _ = gather_indices_path(model, "link0", "link6") # remove the base free joint for IK

    # set initial pose
    data.qpos[q_mask] = np.deg2rad([0,0,-30, 60, -30,0,0])
    mujoco.mj_forward(model, data)

    # visualize
    global viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1 # visualize joints

    # test IK
    test_ik(target_name="ee", goal_name="ik_goal", joint_mask=v_mask,
            pos=[0.3,0.3,0.3], rpy=[0,0,np.pi/4]) # position + orientation

    # test IK with selected elements (position only)
    test_ik(target_name="ee", goal_name="ik_goal", joint_mask=v_mask, element_indices=[0,1,2],
            pos=[0.3,0.2,0.3], rpy=[0,0,np.pi/4]) # position only