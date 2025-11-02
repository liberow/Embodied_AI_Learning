import time

import mujoco.viewer
import ikpy.chain
import transforms3d as tf

def main():
    model = mujoco.MjModel.from_xml_path("../mujoco_menagerie/universal_robots_ur5e/scene.xml")
    data = mujoco.MjData(model)
    my_chain = ikpy.chain.Chain.from_urdf_file("model/ur5e.urdf")

    ee_pos = [-0.13, 0.5, 0.1]
    ee_euler = [3.14, 0, 1.57]
    ref_pos = [0, 0, -1.57, -1.34, 2.65, -1.3, 1.55, 0, 0]  # 9个元素对应9个关节
    ee_orientation = tf.euler.euler2mat(*ee_euler)
    ee_id = model.site("attachment_site").id # 获取末端执行器在模型中的id

    joint_angles = my_chain.inverse_kinematics(ee_pos, ee_orientation, "all", initial_position=ref_pos)
    ctrl = joint_angles[2:-1]  # 跳过前两个固定关节，取6个活动关节
    data.ctrl[:6] = ctrl

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002) # 让动画速度变慢

    ee_pos = data.site_xpos[ee_id]
if __name__ == "__main__":
    main()