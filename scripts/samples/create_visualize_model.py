#!/usr/bin/env -S python3 -i
import mujoco
import mujoco.viewer

# 1) xml with background
xml = r"""
<mujoco model="scene">
  <asset>
    <texture name="whitegrad" type="skybox" builtin="gradient"
             rgb1=".95 .95 .98" rgb2=".85 .85 .90" width="256" height="256"/>
  </asset>
  <worldbody/>
</mujoco>
"""

# 2) generate spec from xml
spec = mujoco.MjSpec()
spec.from_string(xml)

# 3) add model
body = spec.worldbody.add_body(pos=(0,0,0.2))
jnt  = body.add_joint(name="hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=(0,1,0))
geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.05,0.05,0.05), rgba=(0.2,0.4,0.8,1))

# 4) compile spec for model and data
model = spec.compile()
data  = mujoco.MjData(model)

# 5) visualize
viewer = mujoco.viewer.launch_passive(model, data)
viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1 # visualize joints
viewer.sync()