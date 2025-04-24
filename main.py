import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from IMMFilter import CVKF, CTRVKF, IMM_filter

# 设置matplotlib支持LaTeX
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Times New Roman"],
#     "font.size": 15
# })

def add_noise(x_true, noise_std):
    """添加观测噪声"""
    return x_true + np.random.randn(*x_true.shape) * noise_std


def main():
    # 仿真参数设置
    dt = 0.1
    t = np.arange(0.1, 25.1, dt)
    video_save = True
    
    # 生成真实轨迹
    x_truth = []
    p = np.array([0, 0])
    
    for i in range(250):
        if i < 50:
            v = np.array([5, 0])
        elif i < 100:
            r = 5
            v_mag = 10
            w = v_mag / r
            angle = (i-50)/10 * w
            v = np.array([v_mag * np.cos(angle), v_mag * np.sin(angle)])
        elif i < 150:
            v = np.array([3, 5])
        elif i < 200:
            r = 5
            v_mag = 10
            w = v_mag / r
            angle = -(i-50)/10 * w
            v = np.array([v_mag * np.cos(angle), v_mag * np.sin(angle)])
        else:
            v = np.array([3, -5])
            
        p = p + v * dt
        x_truth.append(np.concatenate((p, v)))
    
    x_truth = np.array(x_truth).T
    # z = x_truth[:2,:]  # 观测位置
    z = add_noise(x_truth[:2,:], noise_std=0.05)  # 带噪声的观测
    
    # CV模型初始化
    cv_x_estimation = np.zeros((6, len(t)+1))
    cv_x_estimation[3, 0] = 5
    cv_P = np.eye(6)
    
    cv_parameter = {
        'cv_H': np.array([[1,0,0,0,0,0], [0,1,0,0,0,0]]),
        'cv_Q': np.diag([0.1, 0.1, 1, 0.01, 0.01, 0]),
        'cv_R': np.eye(2) * 1
    }
    
    # CTRV模型初始化
    ctrv_x_estimation = np.zeros((6, len(t)+1))
    ctrv_x_estimation[4, 0] = 1
    ctrv_x_estimation[5, 0] = 0.3
    ctrv_P = np.diag([1,1,1,1,1,1])
    
    ctrv_parameter = {
        'ctrv_H': np.array([[1,0,0,0,0,0], [0,1,0,0,0,0]]),
        'ctrv_Q': np.diag([0.1, 0.1, 1, 0.01, 0.01, 1]),
        'ctrv_R': np.eye(2) * 1.2
    }
    
    # IMM初始化
    IMM_x_estimation = np.zeros((6, len(t)+1))
    IMM_x_estimation[4, 0] = 1
    IMM_x_estimation[5, 0] = 0.1
    
    IMM_parameter = {
        'cv_H': cv_parameter['cv_H'],
        'cv_Q': cv_parameter['cv_Q'],
        'cv_R': cv_parameter['cv_R'],
        'ctrv_H': ctrv_parameter['ctrv_H'],
        'ctrv_Q': ctrv_parameter['ctrv_Q'],
        'ctrv_R': ctrv_parameter['ctrv_R'],
        'P_model': np.array([[0.96, 0.04], [0.04, 0.96]]),
        'mu_weight': np.array([0.9, 0.1]),
        'model': [
            {'x': np.array([0,0,0,1,0,0]), 'P': np.eye(6)},
            {'x': np.array([0,0,0,1,0,0.1]), 'P': np.eye(6)}
        ]
    }
    
    IMM_save = {'mu': []}
    
    # 设置视频保存
    if video_save:
        writer = FFMpegWriter(fps=10)
        fig = plt.figure(figsize=(10, 8))
        with writer.saving(fig, "IMM_tracking.mp4", 100):
            pass
    plt.figure(1, figsize=(12, 8))
    
    # 主循环
    for i in range(len(t)):
        
        # ====== 预测部分 ====== (新增内容)
        # CV模型预测
        cv_x_prediction = np.zeros((6, 11))
        cv_x_prediction[:, 0] = cv_x_estimation[:, i]
        for j in range(10):
            cv_A_j, _, _, _ = CVKF(cv_parameter, z[:,i], cv_x_prediction[:,j], cv_P, dt)
            cv_x_prediction[:, j+1] = cv_A_j @ cv_x_prediction[:,j]
        
        # CTRV模型预测
        ctrv_x_prediction = np.zeros((6, 11))
        ctrv_x_prediction[:, 0] = ctrv_x_estimation[:, i]
        if i == 169:  # Python是0-based索引，对应MATLAB的i==170
            ctrv_x_estimation[5, i] = 0.1
        for j in range(10):
            ctrv_A_j, _, _, _ = CTRVKF(ctrv_parameter, z[:,i], ctrv_x_prediction[:,j], ctrv_P, dt)
            ctrv_x_prediction[:, j+1] = ctrv_A_j @ ctrv_x_prediction[:,j]
        
        # IMM模型预测
        IMM_x_prediction = np.zeros((6, 11))
        IMM_x_prediction[:, 0] = IMM_x_estimation[:, i]
        mu_weight = IMM_parameter['mu_weight']
        for j in range(10):
            cv_A_j, _, _, _ = CVKF(cv_parameter, z[:,i], IMM_x_prediction[:,j], cv_P, dt)
            ctrv_A_j, _, _, _ = CTRVKF(ctrv_parameter, z[:,i], IMM_x_prediction[:,j], ctrv_P, dt)
            IMM_x_prediction[:, j+1] = mu_weight[0] * (cv_A_j @ IMM_x_prediction[:,j]) + \
                                    mu_weight[1] * (ctrv_A_j @ IMM_x_prediction[:,j])
        # ======================
        
        # 预测和更新
        _, cv_x_estimation[:, i+1], cv_P, _ = CVKF(cv_parameter, z[:,i], cv_x_estimation[:,i], cv_P, dt)
        _, ctrv_x_estimation[:, i+1], ctrv_P, _ = CTRVKF(ctrv_parameter, z[:,i], ctrv_x_estimation[:,i], ctrv_P, dt)
        IMM_parameter, IMM_x_estimation[:, i+1], _ = IMM_filter(IMM_parameter, z[:,i], dt)
        IMM_save['mu'].append(IMM_parameter['mu_weight'].copy())
        
        # 可视化
        plt.clf()
        
        # 添加预测轨迹
        plt.plot(cv_x_prediction[0], cv_x_prediction[1], 'b--', linewidth=2, label='CV-KF predicted')
        plt.plot(ctrv_x_prediction[0], ctrv_x_prediction[1], 'g--', linewidth=2, label='CTRV-EKF predicted')
        plt.plot(IMM_x_prediction[0], IMM_x_prediction[1], 'r--', linewidth=2, label='IMM predicted')
        
        # 绘制轨迹
        plt.plot(x_truth[0,:i+1], x_truth[1,:i+1], 'k-', linewidth=2, label='Truth trajectory')
        plt.plot(z[0,:i+1], z[1,:i+1], '.', markersize=15, label='Measurements')
        
        # 绘制估计位置
        plt.plot(cv_x_estimation[0,i], cv_x_estimation[1,i], 'bo', linewidth=2, label='CV-KF estimated')
        plt.plot(ctrv_x_estimation[0,i], ctrv_x_estimation[1,i], 'go', linewidth=2, label='CTRV-EKF estimated')
        plt.plot(IMM_x_estimation[0,i], IMM_x_estimation[1,i], 'ro', linewidth=2, label='IMM estimated')
        
        # 绘制估计轨迹
        plt.plot(cv_x_estimation[0,:i+1], cv_x_estimation[1,:i+1], 'b-', linewidth=2)
        plt.plot(ctrv_x_estimation[0,:i+1], ctrv_x_estimation[1,:i+1], 'g-', linewidth=2)
        plt.plot(IMM_x_estimation[0,:i+1], IMM_x_estimation[1,:i+1], 'r-', linewidth=2)
        
        plt.grid(True)
        plt.axis('equal')
        plt.xlim([-5, 65])
        plt.ylim([-14, 50])
        plt.xlabel(r'$X$/m')
        plt.ylabel(r'$Y$/m')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
        plt.title('IMM Tracking Simulation')
        
        if video_save:
            writer.grab_frame()
        else:
            plt.pause(0.05)
    
    # 绘制最终结果
    plt.figure(1, figsize=(12, 8))
    plt.clf()
    plt.plot(x_truth[0], x_truth[1], 'k-', linewidth=2, label='Truth trajectory')
    plt.plot(z[0], z[1], '.', markersize=15, label='Measurements')
    
    plt.plot(cv_x_estimation[0], cv_x_estimation[1], 'b-', linewidth=2, label='CV-KF estimated')
    plt.plot(ctrv_x_estimation[0], ctrv_x_estimation[1], 'g-', linewidth=2, label='CTRV-EKF estimated')
    plt.plot(IMM_x_estimation[0], IMM_x_estimation[1], 'r-', linewidth=2, label='IMM estimated')
    
    plt.grid(True)
    plt.axis('equal')
    plt.xlim([-5, 65])
    plt.ylim([-14, 50])
    plt.xlabel(r'$X$/m')
    plt.ylabel(r'$Y$/m')
    plt.legend()
    plt.title('Final Tracking Results')
    
    # 绘制误差曲线
    plt.figure(2, figsize=(12, 8))
    plt.subplot(2,1,1)
    plt.plot(t, cv_x_estimation[0,1:]-x_truth[0], 'b-', label='CV-KF')
    plt.plot(t, ctrv_x_estimation[0,1:]-x_truth[0], 'g-', label='CTRV-EKF')
    plt.plot(t, IMM_x_estimation[0,1:]-x_truth[0], 'r-', label='IMM')
    plt.ylabel(r'$p_x$ error (m)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2,1,2)
    plt.plot(t, cv_x_estimation[1,1:]-x_truth[1], 'b-', label='CV-KF')
    plt.plot(t, ctrv_x_estimation[1,1:]-x_truth[1], 'g-', label='CTRV-EKF')
    plt.plot(t, IMM_x_estimation[1,1:]-x_truth[1], 'r-', label='IMM')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$p_y$ error (m)')
    plt.legend()
    plt.grid(True)
    
    # 绘制模型权重
    IMM_mu = np.array(IMM_save['mu'])
    plt.figure(3, figsize=(12, 6))
    plt.plot(t, IMM_mu[:,0], 'b-', label='IMM-CV weight')
    plt.plot(t, IMM_mu[:,1], 'g-', label='IMM-CTRV weight')
    plt.xlabel('Time (s)')
    plt.ylabel('Model weight')
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()