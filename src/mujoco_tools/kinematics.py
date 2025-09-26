import logging
logger = logging.getLogger(__name__)
import mujoco
from mujoco_tools.visualization import *
import numpy as np

def _extend_indices_for_joint(model, jid, qpos_idx, dof_idx):
    """extend qpos_idx and dof_idx depending on joint type by adding indices of multiple DoF joints"""
    jtype = model.jnt_type[jid]
    qadr  = model.jnt_qposadr[jid]
    dadr  = model.jnt_dofadr[jid]
    if jtype == mujoco.mjtJoint.mjJNT_FREE:
        qpos_idx.extend(range(qadr, qadr + 7))
        dof_idx .extend(range(dadr, dadr + 6))
    elif jtype == mujoco.mjtJoint.mjJNT_BALL:
        qpos_idx.extend(range(qadr, qadr + 4))
        dof_idx .extend(range(dadr, dadr + 3))
    else:  # HINGE or SLIDE
        qpos_idx.append(qadr)
        dof_idx .append(dadr)

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
        _extend_indices_for_joint(model, jid, qpos_idx, dof_idx)

    return np.array(qpos_idx), np.array(dof_idx), joint_ids, path

def gather_indices_by_prefix(model, joint_prefix):
    """
    Gather joint indices whose names start with the given prefix.

    Returns:
        qpos_idx (ndarray[int])  : the indices of qpos (sorted, no duplicates)
        dof_idx   (ndarray[int]) : the indices of dof (sorted, no duplicates)
        joint_ids(list[int])     : the IDs of the corresponding joints (unordered)
    """
    joint_ids = []
    qpos_idx, dof_idx = [], []

    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name and name.startswith(joint_prefix):
            joint_ids.append(jid)
            _extend_indices_for_joint(model, jid, qpos_idx, dof_idx)

    return (np.array(sorted(set(qpos_idx))),
            np.array(sorted(set(dof_idx))),
            joint_ids)

def inverse_kinematics(model, data, site_name, goal_name,
                    *, max_iters=200, tol_pos=1e-5, tol_rot=1e-3, damping=1e-3, step_size=1.0,
                    weight_pos=1.0, weight_rot=0.1,
                    joint_mask=None, element_indices=[0,1,2,3,4,5], viewer=None,
                    **kwargs):
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
            goal_pos = data.mocap_pos[mocapid].copy()
            goal_quat = data.mocap_quat[mocapid].copy()
        else:
            raise ValueError(f"mocap body '{goal_name}' is not a mocap")
    else:
        raise ValueError(f"body '{goal_name}' not found")

    # joint mask
    if joint_mask is None:
        joint_mask = np.ones(model.nv, dtype=bool)

    # ik loop
    ik_result = False
    num_elements = len(element_indices)
    if num_elements == 0:
        raise ValueError("element_indices is empty")
    pos_indices = sorted(set(range(0, 3)) & set(element_indices)) # select position indices
    rot_indices = sorted(set(range(3, 6)) & set(element_indices)) # select rotation indices

    logger.info(f"IK start")
    logger.info(f" site_name '{site_name}' (site_id {site_id}), goal_name '{goal_name}' (mocap id {mocapid})")
    logger.info(f" goal_pos {goal_pos}, goal quat {goal_quat}")
    logger.info(f" joint_mask {joint_mask}")
    logger.info(f" element_indices {element_indices}")
    logger.info(f" pos_indices {pos_indices}, rot_indices {rot_indices}")
    logger.info(f" num_elements {num_elements}")
    logger.info(f" max_iters {max_iters}, tol_pos {tol_pos}, tol_rot {tol_rot}, damping {damping}, step_size {step_size}")
    logger.info(f" weight_pos {weight_pos}, weight_rot {weight_rot}")

    for it in range(max_iters):
        mujoco.mj_forward(model, data)

        # error
        err = np.zeros(6)
        # error to goal pos
        cur_pos = data.site_xpos[site_id].copy()
        err[0:3] = goal_pos - cur_pos
        # error to goal rot
        cur_rot = data.site_xmat[site_id].copy()
        cur_quat = np.zeros(4); mujoco.mju_mat2Quat(cur_quat, cur_rot.reshape(-1))
        # # use quat2Vel
        # qinv = np.zeros(4); mujoco.mju_negQuat(qinv, cur_quat) # conj
        # q_err = np.zeros(4); mujoco.mju_mulQuat(q_err, goal_quat, qinv) # rotation error (world): q_err = q_goal * conj(q_cur)
        # if q_err[0] < 0: # ensure the same side hemisphere for shortest path
        #     q_err = -q_err
        # mujoco.mju_quat2Vel(err[3:6], q_err, 1.0) # convert quat diff -> angular velocity (dt=1): world-aligned
        # use subQuat
        mujoco.mju_subQuat(err[3:6], goal_quat, cur_quat) # rotation error (cur_quat * quat(err) = goal_quat)
        err[3:6] = cur_rot.reshape((3,3)) @ err[3:6] # subQuat is in local frame, convert to world-aligned
        # check convergence
        if np.linalg.norm(err[pos_indices]) < tol_pos and np.linalg.norm(err[rot_indices]) < tol_rot:
            ik_result = True
            break

        # jacobi matrix
        Jp = np.zeros((3, model.nv))
        Jr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, Jp, Jr, site_id)
        J = np.vstack((Jp, Jr))

        # extract used DoF
        J = J[element_indices,:][:, joint_mask] # select rows and columns

        # DLS: dq = (J^T W J + λ^2 I)^{-1} * J^T W err
        W = np.diag([weight_pos]*3 + [weight_rot]*3)[element_indices,:][:,element_indices]
        lam2 = damping * damping
        H = J.T @ W @ J + lam2 * np.eye(J.shape[1])
        g = J.T @ W @ err[element_indices]
        dq_used = np.linalg.solve(H, g * step_size)

        # extend to full DoF
        dq = np.zeros(model.nv)
        dq[joint_mask] = dq_used

        # update qpos
        mujoco.mj_integratePos(model, data.qpos, dq, 1.0)

        logger.debug(f"IK iter {it}: pos err {np.linalg.norm(err[pos_indices])}, rot err {np.linalg.norm(err[rot_indices])}")

    # visualize target and goal
    if viewer is not None:
        # target
        draw_frame(viewer, data.site_xpos[site_id], data.site_xmat[site_id].reshape((3,3)), scale=0.15, width=0.01)
        # goal
        goal_rot = np.zeros((9,))
        mujoco.mju_quat2Mat(goal_rot, goal_quat)
        draw_frame(viewer, goal_pos, goal_rot.reshape((3,3)), scale=0.2, width=0.01, alpha=0.3)

    logger.info(f"IK iter {it}: pos err {np.linalg.norm(err[pos_indices])}, rot err {np.linalg.norm(err[rot_indices])}")

    return ik_result, it
