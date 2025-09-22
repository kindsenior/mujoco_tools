#!/usr/bin/env -S python3 -i
import numpy as np
import mujoco
import mujoco.viewer
from mujoco_tools.kinematics import *

def test_ik():
    # create model
    from mujoco_tools.modeling import sample_manipulator
    spec = mujoco.MjSpec()
    sample_manipulator(spec, free_joint=True) # add a free joint to the base link
    global model, data
    model = spec.compile()
    data = mujoco.MjData(model)

    q_mask, v_mask, _, _ = gather_indices_path(model, "link0", "link6") # remove the base free joint for IK

    # set initial pose
    data.qpos[q_mask] = np.deg2rad([0,0,-30, 60, -30,0,0])
    mujoco.mj_forward(model, data)

    # add the goal mocap
    goal_body = spec.worldbody.add_body(name="ik_goal", mocap=True) # set to mocap body

    # rebuild
    model = spec.compile()
    data = mujoco.MjData(model)

    # set target
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ik_goal")
    mid = model.body_mocapid[bid] # convert body id to mocap id (mocaps also belong to body class)
    data.mocap_pos[mid] = [0.3, 0.3, 0.3]
    data.mocap_quat[mid] = [1, 0, 0, 0]

    # IK
    ok, iters = inverse_kinematics(model, data, site_name="ee", goal_name="ik_goal", joint_mask=v_mask,
                                max_iters=200, tol_pos=1e-6, damping=1e-3, step_size=0.7)
    print("IK success:", ok, "iters:", iters)

    global viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1 # visualize joints
    viewer.sync()

if __name__ == "__main__":
    test_ik()