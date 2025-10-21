#!/usr/bin/env -S python3 -i
import mujoco
import mujoco.viewer
from mujoco_tools.modeling import sample_robot_biped
from mujoco_tools.visualization import *
import numpy as np
np.set_printoptions(linewidth=110) # 6 columns per one line

# 1) create robot model
spec = mujoco.MjSpec()
spec.from_string(common_xml)
# sample_robot_biped(spec, free_joint=True, contype=2, conaffinity=1)
sample_robot_biped(spec, free_joint=True, contype=1, conaffinity=1)
model = spec.compile()
data = mujoco.MjData(model)

# visualize
viewer = mujoco.viewer.launch_passive(model, data)
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
viewer.sync()

# 2) set states
data.qpos = [0,0,0.55,1,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0]
# data.qpos = [0,0,0.55,1,0,0,0, 0,0,-0.3,0.6,-0.3,0, 0,0,-0.3,0.6,-0.3,0]
data.qvel = [0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0]

# 3) forward kinematics
# mujoco.mj_forward(model, data)
data.qacc = [0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0] # set qacc after mj_forward
mujoco.mj_kinematics(model, data)

# 4) apply external force to end effector
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_ee")
body_id = model.site_bodyid[site_id]
p_world = data.site_xpos[site_id].copy() # application point in world frame
F_world = -model.opt.gravity * np.sum(model.body_mass)   # force [N]   (robot weight)
T_world = np.cross(data.subtree_com[0]-p_world, F_world) # moment [Nm] (The root link id is 0)
print("Apply external force to the end effector")
print(f" site_id={site_id}, body_id={body_id}")
print(f" p_world={p_world}, F_world={F_world}, T_world={T_world}")

# clear external forces (accumulated)
data.qfrc_applied[:] = 0.0
data.xfrc_applied[:] = 0.0

# generalized forces for external forces
mujoco.mj_applyFT(model, data, F_world, T_world, p_world, body_id, data.qfrc_applied)

# 5) inverse dynamics
mujoco.mj_inverse(model, data)

# 6) result
print("nv=", model.nv)
print("qacc        =", data.qacc.copy())
Mqacc = np.zeros(model.nv)
mujoco.mj_mulM(model, data, Mqacc, data.qacc)
print("M qacc =\n", Mqacc)
print("qfrc_bias   =\n", data.qfrc_bias.copy())
print("qfrc_passive=\n", data.qfrc_passive.copy())
print("qfrc_constraint=\n", data.qfrc_constraint.copy())
print("qfrc_applied=\n", data.qfrc_applied.copy()) # the sum of J^T f
# qfrc_inverse and the output torque should be the same values (but somehow applied forces are not included)
print("qfrc_inverse=\n", data.qfrc_inverse.copy())

print("\n")
print("Check the error of inverse dynamics")
tau_jnt = Mqacc + data.qfrc_bias - data.qfrc_applied - data.qfrc_passive - data.qfrc_constraint
print("M qacc + bias - applied - passive - constraints=\n", tau_jnt)
print("M qacc + bias - passive - constraints=\n", (tau_jnt + data.qfrc_applied))
# difference
print("difference")
print("qfrc_inverse - (M qacc + bias - passive - constraints)=\n", data.qfrc_inverse - (tau_jnt+ data.qfrc_applied))
print("qfrc_inverse - (M qacc + bias - applied - passive - constraints)=\n", data.qfrc_inverse - tau_jnt)

# visualize
viewer.user_scn.ngeom = 0 # reset the start index of geometries
# display end effector forces
draw_arrow(viewer, data.site_xpos[model.site("right_ee").id], F_world, scale=0.01, width=0.01)
# display joint torques
draw_torque(viewer, model, data, max_torque=5)

viewer.sync()