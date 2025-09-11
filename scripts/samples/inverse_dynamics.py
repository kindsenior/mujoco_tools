#!/usr/bin/env -S python3 -i
import modeling_robot_configuration
import mujoco
from mujoco_tools.visualization import *
import numpy as np

# 1) create robot model
spec = mujoco.MjSpec()
spec.from_string(common_xml)
modeling_robot_configuration.sample_manipulator(spec)
model = spec.compile()
data = mujoco.MjData(model)

# visualize
viewer = mujoco.viewer.launch_passive(model, data)
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
viewer.sync()

# 2) set states
data.qpos = [np.pi/4, 0, 0, np.pi/2, 0, 0, 0]
data.qvel = np.array([np.pi/2, 0, 0, 0, 0, 0, 0])

# 3) forward kinematics
mujoco.mj_forward(model, data)
data.qacc = np.array([0, 0, 0, np.pi/2, 0, 0, 0]) # set qacc after mj_forward

# 4) apply external force to end effector
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
body_id = model.site_bodyid[site_id]
p_world = data.site_xpos[site_id].copy()     # 作用点（ワールド座標）
F_world = np.array([0.0, 0.0,  50.0])        # 力[N]
T_world = np.array([0.0, 0.0,  0.0])         # モーメント[Nm]

# clear external forces (accumulated)
data.qfrc_applied[:] = 0.0
data.xfrc_applied[:] = 0.0

# generalized forces for external forces
mujoco.mj_applyFT(model, data, F_world, T_world, p_world, body_id, data.qfrc_applied)
print(f"Applied forces (joint space): {data.qfrc_applied}")

# 5) inverse dynamics
mujoco.mj_inverse(model, data)

# 6) result
print("nv=", model.nv)
print("qacc        =", data.qacc.copy())
Mqacc = np.zeros(model.nv)
mujoco.mj_mulM(model, data, Mqacc, data.qacc)
print("M qacc =", Mqacc)
print("qfrc_bias   =", data.qfrc_bias.copy())
print("qfrc_applied=", data.qfrc_applied.copy()) # the sum of J^T f
# qfrc_inverse and the output torque should be the same values (but somehow applied forces are not included)
print("qfrc_inverse=\n", data.qfrc_inverse.copy())
tau_jnt = Mqacc + data.qfrc_bias - data.qfrc_applied - data.qfrc_passive - data.qfrc_constraint
print("M qacc + bias - applied - passive - constraints=\n", tau_jnt)
print("M qacc + bias - passive - constraints=\n", (tau_jnt + data.qfrc_applied))

# visualize
viewer.user_scn.ngeom = 0 # reset the start index of geometries
# display end effector forces
draw_arrow(viewer, data.site_xpos[model.site("ee").id], F_world, scale=0.01, width=0.01)
# display joint torques
for jid in range(model.njnt):
    if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
        continue  # skip non-hinge joints

    # scale joint torques
    arc_angle = tau_jnt[model.jnt_dofadr[jid]] * 2*np.pi/100.0

    draw_arc(viewer, data.xanchor[jid], data.xaxis[jid], arc_angle,
             radius=0.05, nseg=16, width=0.006)

viewer.sync()