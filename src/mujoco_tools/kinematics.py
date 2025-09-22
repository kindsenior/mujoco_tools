import mujoco
import numpy as np

def gather_indices_path(model, start_body_name, end_body_name):
    """gather joint indices on the path from start to end body"""
    # get body ids
    bid_start = model.body(start_body_name).id
    bid_end   = model.body(end_body_name).id

    # search path from start to end by ascending parents
    path = []
    bid = bid_end
    for _ in range(model.nbody + 1): # prevent infinite loop
        path.append(bid)
        if bid == bid_start:
            break
        parent = model.body_parentid[bid]
        if parent == bid or bid == 0: # reached the world(parent=bid=0) or the abnormal cases(parent is self)
            raise ValueError(
                f"body '{start_body_name}' is not an ancestor of body '{end_body_name}'"
            )
        bid = parent
    else:
        raise RuntimeError("Ascending parents did not converge (model inconsistency?)")
    path = path[::-1] # reverse to get from start to end

    # gather joint ids on the path
    joint_ids = []
    for body in path:
        adr = model.body_jntadr[body]
        num = model.body_jntnum[body] # body_jntnum returns the number of joints between the body and its parent
        for j in range(num):
            joint_ids.append(adr + j)

    # gather qpos_idx, dof_idx depending on joint type
    qpos_idx, dof_idx = [], []
    for jid in joint_ids:
        jtype = model.jnt_type[jid]
        qpos_adr  = model.jnt_qposadr[jid]
        dof_adr  = model.jnt_dofadr[jid]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            qpos_idx += list(range(qpos_adr, qpos_adr+7))
            dof_idx  += list(range(dof_adr, dof_adr+6))
        elif jtype == mujoco.mjtJoint.mjJNT_BALL:
            qpos_idx += list(range(qpos_adr, qpos_adr+4))
            dof_idx  += list(range(dof_adr, dof_adr+3))
        else:  # hinge or slide
            qpos_idx.append(qpos_adr)
            dof_idx.append(dof_adr)

    return np.array(qpos_idx), np.array(dof_idx), joint_ids, path

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
