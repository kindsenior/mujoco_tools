#!/usr/bin/env -S python3 -i
import numpy as np
import mujoco
import mujoco.viewer

def inverse_kinematics(model, data, site_name, goal_name,
                    max_iters=200, tol_pos=1e-5, damping=1e-3, step_size=1.0,
                    joint_mask=None):
    """
    - site_name: the target site for movement
    - goal_name: the goal site for movement
    - joint_mask: specify the DoF to use with True/False (length model.nv). If not specified, all movable DoF will be used.
    """
    # the target site for movement
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"site '{site_name}' not found")

    # the goal mocap
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, goal_name)
    if bid >= 0:
        mocapid = model.body_mocapid[bid]  # -1 when not mocap
        if mocapid >= 0:
            target_pos = data.mocap_pos[mocapid].copy()
            target_quat = data.mocap_quat[mocapid].copy()

    # joint mask
    if joint_mask is None:
        joint_mask = np.ones(model.nv, dtype=bool)

    # ik loop
    for it in range(max_iters):
        mujoco.mj_forward(model, data)

        # error to target pos
        cur_pos = data.site_xpos[site_id].copy()
        err_pos = target_pos - cur_pos
        if np.linalg.norm(err_pos) < tol_pos:
            return True, it
        # error to target rot
        cur_rot = data.site_xmat[site_id].copy()
        cur_quat = np.zeros(4)
        mujoco.mju_mat2Quat(cur_quat, cur_rot.reshape(-1))
        err_rot = np.zeros(3)
        mujoco.mju_subQuat(err_rot, target_quat, cur_quat)
        err = np.hstack((err_pos, err_rot))

        # jacobi matrix
        Jp = np.zeros((3, model.nv))
        Jr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, Jp, Jr, site_id)
        J = np.vstack((Jp, Jr))

        # extract used DoF
        J = J[:, joint_mask] # 3 x nv_used

        # DLS: dq = J^T (J J^T + λ^2 I)^{-1} * err
        lam2 = damping * damping
        A = J @ J.T + lam2 * np.eye(6)
        dq_used = J.T @ np.linalg.solve(A, err * step_size)

        # extend to full DoF
        dq = np.zeros(model.nv)
        dq[joint_mask] = dq_used

        # update qpos
        mujoco.mj_integratePos(model, data.qpos, dq, 1.0)

    return False, max_iters

def test_ik():
    # create model
    from mujoco_tools.modeling import sample_manipulator
    spec = mujoco.MjSpec()
    sample_manipulator(spec)
    global model, data
    model = spec.compile()
    data = mujoco.MjData(model)

    data.qpos = np.deg2rad([0,0,-30, 60, -30,0,0])
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
    ok, iters = inverse_kinematics(model, data, site_name="ee", goal_name="ik_goal",
                                max_iters=200, tol_pos=1e-6, damping=1e-3, step_size=0.7)
    print("IK success:", ok, "iters:", iters)

    global viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1 # visualize joints
    viewer.sync()

if __name__ == "__main__":
    test_ik()