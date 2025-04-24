import numpy as np
from scipy.linalg import inv

def IMM_filter(parameter, z, dt):
    """
    IMM滤波算法的Python实现
    
    参数:
    parameter - 包含模型参数、权重等的字典
    z - 当前时刻的观测值
    dt - 时间步长
    
    返回:
    parameter - 更新后的参数
    x - 融合后的状态估计
    P - 融合后的协方差矩阵
    """
    
    # 从parameter中提取参数
    P_model = parameter['P_model']
    mu_weight = parameter['mu_weight']
    model = parameter['model']
    
    # 1. 计算混合概率
    c_bar = np.zeros(2)
    for i in range(2):
        for j in range(2):
            c_bar[i] += P_model[i,j] * mu_weight[j]
    
    # 计算混合权重
    mu = np.zeros((2,2))
    for i in range(2):
        mu[i,:] = P_model[i,:] * mu_weight[i] / c_bar
    
    mu = mu.T
    
    # 保存预测前的状态和协方差
    for i in range(2):
        model[i]['x_pre'] = model[i]['x'].copy()
        model[i]['P_pre'] = model[i]['P'].copy()
    
    # 2. 交互/混合
    for i in range(2):
        model[i]['x'] = np.zeros(6)
        for j in range(2):
            model[i]['x'] += mu[i,j] * model[j]['x_pre']
        
        # 归一化处理(假设4-5元素是方向向量)
        norm_val = np.linalg.norm(model[i]['x'][3:5])
        if norm_val > 0:
            model[i]['x'][3:5] = model[i]['x'][3:5] / norm_val
    
    for i in range(2):
        model[i]['P'] = np.zeros((6,6))
        for j in range(2):
            delta_x = model[i]['x'] - model[j]['x_pre']
            model[i]['P'] += mu[i,j] * (model[j]['P_pre'] + np.outer(delta_x, delta_x))
    
    # 3. 模型条件滤波
    # 假设CVKF和CTRVKF是单独的函数
    _, model[0]['x'], model[0]['P'], cv_x_pred = CVKF(parameter, z, model[0]['x'], model[0]['P'], dt)
    _, model[1]['x'], model[1]['P'], ctrv_x_pred = CTRVKF(parameter, z, model[1]['x'], model[1]['P'], dt)
    
    # 计算新息
    model[0]['v'] = z - parameter['cv_H'] @ cv_x_pred
    model[1]['v'] = z - parameter['ctrv_H'] @ ctrv_x_pred
    
    # 计算新息协方差
    model[0]['S'] = parameter['cv_H'] @ model[0]['P'] @ parameter['cv_H'].T + parameter['cv_R']
    model[1]['S'] = parameter['ctrv_H'] @ model[1]['P'] @ parameter['ctrv_H'].T + parameter['ctrv_R']
    ## 以上两个都可以在滤波计算的过程中保存下来，就不必再算一遍了
    
    
    # 计算模型似然
    Hat = np.zeros(2)
    Hat[0] = 1/(np.sqrt(2)*np.sqrt(np.pi)*np.sqrt(np.linalg.det(model[0]['S']))) * \
             np.exp(-0.5 * model[0]['v'].T @ inv(model[0]['S']) @ model[0]['v'])
    
    Hat[1] = 1/(np.sqrt(2)*np.sqrt(np.pi)*np.sqrt(np.linalg.det(model[1]['S']))) * \
             np.exp(-0.5 * model[1]['v'].T @ inv(model[1]['S']) @ model[1]['v'])
    
    # 4. 更新模型概率
    c = np.sum(Hat * c_bar)
    mu_weight = Hat * c_bar / c
    
    # 5. 估计组合
    x = model[0]['x'] * mu_weight[0] + model[1]['x'] * mu_weight[1]
    
    P = (model[0]['P'] + np.outer(x - model[0]['x'], x - model[0]['x'])) * mu_weight[0] + \
        (model[1]['P'] + np.outer(x - model[1]['x'], x - model[1]['x'])) * mu_weight[1]
    
    # 更新parameter
    parameter['mu_weight'] = mu_weight
    parameter['model'] = model
    
    return parameter, x, P

import numpy as np

def CVKF(cv_parameter, z, x, cv_P, dt):
    """
    恒定速度(CV)模型卡尔曼滤波的Python实现
    
    参数:
    cv_parameter - 包含CV模型参数的字典
    z - 当前时刻的观测值
    x - 当前状态估计
    cv_P - 当前协方差矩阵
    dt - 时间步长
    
    返回:
    cv_A - 状态转移矩阵
    x - 更新后的状态估计
    cv_P - 更新后的协方差矩阵
    x_pred - 预测状态
    """

    # 从参数中提取需要的值
    cv_Q = cv_parameter['cv_Q']
    cv_R = cv_parameter['cv_R']
    cv_H = cv_parameter['cv_H']
    
    # 提取状态变量
    v = x[2]  # Python是0-based索引
    cos_i = x[3]
    sin_i = x[4]
    
    # 构建状态转移矩阵A
    cv_A = np.array([
        [0, 0, cos_i, v, 0, 0],
        [0, 0, sin_i, 0, v, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    
    cv_A = np.eye(6) + cv_A * dt
    
    # 预测步骤
    x_pred = cv_A @ x
    cv_P = cv_A @ cv_P @ cv_A.T + cv_Q
    
    # 更新步骤
    S = cv_H @ cv_P @ cv_H.T + cv_R
    K = cv_P @ cv_H.T @ np.linalg.inv(S)
    
    x = x_pred + K @ (z - cv_H @ x_pred)
    cv_P = (np.eye(6) - K @ cv_H) @ cv_P
    
    # 归一化处理
    norm_val = np.linalg.norm(x[3:5])
    if norm_val > 0:
        x[3:5] = x[3:5] / norm_val
        x[2] = x[2] * norm_val  # 调整速度
    
    return cv_A, x, cv_P, x_pred

import numpy as np

def CTRVKF(ctrv_parameter, z, x, ctrv_P, dt):
    """
    恒定转向率和速度(CTRV)模型扩展卡尔曼滤波的Python实现
    
    参数:
    ctrv_parameter - 包含CTRV模型参数的字典
    z - 当前时刻的观测值
    x - 当前状态估计
    ctrv_P - 当前协方差矩阵
    dt - 时间步长
    
    返回:
    ctrv_A - 状态转移矩阵
    x - 更新后的状态估计
    ctrv_P - 更新后的协方差矩阵
    x_pred - 预测状态
    """
    
    # 从参数中提取需要的值
    ctrv_H = ctrv_parameter['ctrv_H']
    ctrv_Q = ctrv_parameter['ctrv_Q']
    ctrv_R = ctrv_parameter['ctrv_R']
    
    # 提取状态变量
    v_i = x[2]  # 速度
    cos_i = x[3]  # 方向的cos值
    sin_i = x[4]  # 方向的sin值
    w_i = x[5]   # 转向率
    
    # 处理w_i接近0的情况以避免除以0
    if abs(w_i) < 1e-5:
        w_i = 1e-5 if w_i >= 0 else -1e-5
    
    # 计算状态转移矩阵元素
    wdt = w_i * dt
    cos_wdt = np.cos(wdt)
    sin_wdt = np.sin(wdt)
    
    # 计算A矩阵的各元素
    A_13 = (sin_i*(cos_wdt-1) + cos_i*sin_wdt) / w_i
    A_14 = v_i * sin_wdt / w_i
    A_15 = v_i * (cos_wdt-1) / w_i
    A_16 = (-v_i/w_i**2 * (sin_i*(cos_wdt-1) + cos_i*sin_wdt) + 
            v_i*dt/w_i * (-sin_i*sin_wdt + cos_i*cos_wdt))
    
    A_23 = (cos_i*(1 - cos_wdt) + sin_i*sin_wdt) / w_i
    A_25 = v_i * sin_wdt / w_i
    A_24 = -v_i * (cos_wdt-1) / w_i
    A_26 = (-v_i/w_i**2 * (cos_i*(1 - cos_wdt) + sin_i*sin_wdt) + 
            v_i*dt/w_i * (sin_i*cos_wdt + cos_i*sin_wdt))
    
    # 构建状态转移矩阵
    ctrv_A = np.array([
        [1, 0, A_13, A_14, A_15, A_16],
        [0, 1, A_23, A_24, A_25, A_26],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, cos_wdt, -sin_wdt, -cos_i*dt*sin_wdt - sin_i*dt*cos_wdt],
        [0, 0, 0, sin_wdt, cos_wdt, -sin_i*dt*sin_wdt + cos_i*dt*cos_wdt],
        [0, 0, 0, 0, 0, 1]
    ])
    
    # 预测步骤
    x_pred = ctrv_A @ x
    ctrv_P = ctrv_A @ ctrv_P @ ctrv_A.T + ctrv_Q
    
    # 更新步骤
    S = ctrv_H @ ctrv_P @ ctrv_H.T + ctrv_R
    K = ctrv_P @ ctrv_H.T @ np.linalg.inv(S)
    
    x = x_pred + K @ (z - ctrv_H @ x_pred)
    ctrv_P = (np.eye(6) - K @ ctrv_H) @ ctrv_P
    
    # 归一化处理
    norm_val = np.linalg.norm(x[3:5])
    if norm_val > 0:
        x[3:5] = x[3:5] / norm_val
        x[2] = np.abs(x[2]) * norm_val  # 保持速度符号
    
    return ctrv_A, x, ctrv_P, x_pred