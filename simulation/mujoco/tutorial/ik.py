import ikpy.chain
import ikpy.utils.plot as plot_utils
import matplotlib.pyplot as plt
import transforms3d as tf

def main():
    my_chain = ikpy.chain.Chain.from_urdf_file("model/ur5e.urdf")
    ee_pos = [-0.13, 0.5, 0.1] # 末端执行器在世界坐标系下的位置 的 x,y,z 坐标
    ee_euler = [3.14, 0, 1.57] # 末端执行器在欧拉角姿态
    ee_orientation = tf.euler.euler2mat(*ee_euler) # 将欧拉角姿态转换为旋转矩阵, 欧拉角有偏差
    ref_pos = [0, 0, -1.57, -1.34, 2.65, -1.3, 1.55, 0, 0]  # 初始猜测解，ik 有多个解，猜测解是尽可能让它收敛到我想要的解

    fig, ax = plot_utils.init_3d_figure()
    my_chain.plot(
        joints = my_chain.inverse_kinematics(
            target_position=ee_pos, 
            target_orientation=ee_orientation, 
            orientation_mode="all", # 要围绕x, y, z轴旋转，所以是all
            initial_position=ref_pos,
        ), 
        ax=ax,
    )
    plt.show()

if __name__ == "__main__":
    main()