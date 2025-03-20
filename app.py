from flask import Flask, render_template, request, jsonify
import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import statsmodels.api as sm
from PIL import Image

# 初始化MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# 关键点索引
NOSE, LEFT_EAR, RIGHT_EAR, LEFT_MOUTH, RIGHT_MOUTH = 0, 7, 8, 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# 头部和脚部的 Mediapipe 关键点索引
head_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 头部 10 个点
foot_indices = [27, 28, 29, 30, 31, 32]  # 脚部 6 个点

def euclidean_dist(x1, y1, x2, y2):
    """计算两个点之间的欧几里得距离"""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def max_head_to_foot_distance(cx_list, cy_list):
    max_distance = 0
    
    # 遍历所有头部和脚部点，计算欧几里得距离
    for h in head_indices:
        for f in foot_indices:
            x1, y1 = cx_list[h], cy_list[h]
            x2, y2 = cx_list[f], cy_list[f]
            
            # 计算欧几里得距离
            distance = euclidean_dist(x1, y1, x2, y2)
            
            # 记录最大距离
            if distance > max_distance:
                max_distance = distance
    
    return max_distance

def calculate_body_height(cx_list, cy_list):
    """计算人体的完整高度，包括头顶和脚底，并返回 body_height"""

    # 计算肩膀中点
    shoulder_mid_x = (cx_list[LEFT_SHOULDER] + cx_list[RIGHT_SHOULDER]) / 2
    shoulder_mid_y = (cy_list[LEFT_SHOULDER] + cy_list[RIGHT_SHOULDER]) / 2

    # 计算鼻子到肩膀中点的距离
    head_to_shoulder = euclidean_dist(cx_list[NOSE], cy_list[NOSE], shoulder_mid_x, shoulder_mid_y)

    # 计算左侧和右侧身高（使用欧几里得距离）
    def compute_side_height(shoulder, hip, knee, ankle):
        parts = [
            (cx_list[shoulder], cy_list[shoulder], cx_list[hip], cy_list[hip]),  # 肩到髋
            (cx_list[hip], cy_list[hip], cx_list[knee], cy_list[knee]),          # 髋到膝
            (cx_list[knee], cy_list[knee], cx_list[ankle], cy_list[ankle])       # 膝到踝
        ]
        return sum(euclidean_dist(x1, y1, x2, y2) for x1, y1, x2, y2 in parts)

    left_height = compute_side_height(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    right_height = compute_side_height(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)

    # 计算人体总高度
    body_height = max(left_height, right_height)
    body_height += head_to_shoulder  # 加上鼻子到肩膀的高度
    body_height += 1.7 * head_to_shoulder  # 加上头顶到鼻子的高度
    body_height += 0.05 * body_height # 加上脚的高度

    max_distance = max_head_to_foot_distance(cx_list, cy_list)

    return max(body_height, max_distance)

def calculate_head_y(cy_list, body_height):
    """计算头部的归一化高度（相对于地面高度），处理 None 值"""
    
    valid_y_values = [y for y in cy_list if y is not None]
    ground_y = max(valid_y_values)

    head_y = (ground_y - cy_list[NOSE]) / body_height

    return head_y

def analyze_orientation(cx_list, cy_list):
    # 计算鼻子-左耳、鼻子-右耳的角度
    dx_left = cy_list[NOSE] - cy_list[LEFT_EAR]
    dy_left = cx_list[NOSE] - cx_list[LEFT_EAR]

    dx_right = cy_list[NOSE] - cy_list[RIGHT_EAR]
    dy_right = cx_list[NOSE] - cx_list[RIGHT_EAR]

    # 计算左右耳相对鼻子的角度（转换为角度制）
    angle_left = abs(np.degrees(np.arctan2(dy_left, dx_left)))
    angle_right = abs(np.degrees(np.arctan2(dy_right, dx_right)))

    # 计算平均角度
    avg_angle = (angle_left + angle_right) / 2

    dy_mouth = (cy_list[LEFT_MOUTH] + cy_list[RIGHT_MOUTH])/2

    # 根据角度分类姿态
    if 40 <= avg_angle <= 100:
        if cy_list[NOSE] > dy_mouth:
            return "down"
        else:
            return "neutral"
    elif 0 <= avg_angle <= 40:
        return "down" 
    elif 100 <= avg_angle <= 180:
        return "up"
    else:
        return "unknown"

def process_pose(image):
    """处理人体姿态，返回关键点坐标，如果数据缺失则直接返回 'Insufficient data'"""

    # 颜色转换 (BGR -> RGB)
    img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 进行姿态检测
    results = pose.process(img_RGB)
    
    # 确保 `pose_landmarks` 存在
    if not results.pose_landmarks:
        return "Insufficient data"

    # 获取图像宽高
    h, w = image.shape[:2]

    # 提取所有 33 个关键点的像素坐标
    cx_list = [int(lm.x * w) for lm in results.pose_landmarks.landmark]
    cy_list = [int(lm.y * h) for lm in results.pose_landmarks.landmark]

    # 需要的关键点索引
    required_points = [NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER,
                       LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]

    # 关键点完整性检查
    if any(cx_list[p] is None or cy_list[p] is None for p in required_points):
        return "Insufficient data"

    return cx_list, cy_list

def process_frame(image):
    """处理单帧图像，并返回 JSON 记录"""

    if image is None:
        return None
    
    frame_data = {
        "body_height": None,
        "orientation": None,
        "head_y": None
    }

    pose_data = process_pose(image)
    if pose_data != "Insufficient data":
        cx_list, cy_list = pose_data

        body_height = calculate_body_height(cx_list, cy_list)
        frame_data.update({"body_height": body_height})

        orientation = analyze_orientation(cx_list, cy_list)
        frame_data.update({"orientation": orientation})

        head_y = calculate_head_y(cy_list, body_height)
        frame_data.update({"head_y": head_y})
    
    return frame_data


def generate_video(input_video):
    """读取视频，每 2 帧读取 1 帧，并返回所有帧的数据列表"""

    try:
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError("错误: 无法打开视频文件！")
    except Exception as e:
        print(f"视频读取失败: {e}")
        return None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_data_list = []  # 存储帧数据
    last_frame_data = None  # 记录上一帧的数据
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    with tqdm(total=total_frames) as pbar:
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame is None:
                break

            if frame_idx % 2 == 0:  # 每 2 帧处理 1 帧
                try:
                    frame_data = process_frame(frame)
                    last_frame_data = frame_data  # 更新上一帧数据
                except Exception as e:
                    print(f"处理帧时出错: {e}")
                    frame_data = None
            else:
                frame_data = last_frame_data  # 复制上一帧的数据

            frame_data_list.append(frame_data)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()
    
    return frame_data_list, fps

def extract_data_from_frame_list(frame_data_list):
    """从 frame_data_list 中提取 4 个独立的数组"""
    body_height_list = []
    orientation_list = []
    head_y_list = []

    for frame_data in frame_data_list:
        if frame_data is None:
            # 如果 frame_data 是 None，则添加 None 占位符
            body_height_list.append(None)
            orientation_list.append(None)
            head_y_list.append(None)
        else:
            # 从每一个 frame_data 提取对应数据
            body_height_list.append(frame_data.get("body_height"))
            orientation_list.append(frame_data.get("orientation"))
            head_y_list.append(frame_data.get("head_y"))
    
    return body_height_list, orientation_list, head_y_list

def smooth_stable_data(orientation, window_size=10, consensus_ratio=0.8):
    """
    平滑数据，移除噪音，使 `people_counts` 和 `orientation` 更稳定。
    
    该方法使用滑动窗口计算最常见值，并在比例达到 `consensus_ratio` 时替换当前值，
    以减少噪声的影响，使数据更平滑。
    
    参数:
        people_counts (list[int]): 每一帧检测到的人数数据。
        orientation (list[str]): 每一帧的面部朝向信息。
        motion_states (list[str]): 每一帧的运动状态 ('static' 或 'dynamic')。
        window_size (int): 滑动窗口的大小，决定平滑时考虑的帧数（默认 10）。
        consensus_ratio (float): 认定最常见值的比例，若达到该比例则采用最常见值（默认 0.8）。

    返回:
        tuple: 包含平滑后的数据：
            - filtered_people_counts (list[int]): 平滑后的人数数据。
            - filtered_orientation (list[str]): 平滑后的面部朝向数据。
            - filtered_motion_states (list[str]): 平滑后的运动状态数据。
    """
    
    # 复制原始数据，避免修改输入列表
    filtered_orientation = orientation[:]

    # 遍历所有帧数据
    for i in range(len(orientation)):
        # 定义滑动窗口的范围
        start, end = max(0, i - window_size), min(len(orientation), i + window_size)
        # 计算滑动窗口内的最常见值
        most_common_orientation = max(set(orientation[start:end]), key=orientation[start:end].count)
        # 计算最常见值的占比
        orientation_consensus = orientation[start:end].count(most_common_orientation) / (end - start)
        # 如果最常见值的比例超过 `consensus_ratio`，则采用它，否则保持原值
        filtered_orientation[i] = most_common_orientation if orientation_consensus >= consensus_ratio else orientation[i]

    return filtered_orientation

def first_orientation_segments(orientation, body_height, head_y, fps):

    orient_segments = []
    current_orient, start_frame = None, None

    # 预处理，将 None 变成 'Invalid'
    orientation = ['Invalid' if orientation is None else orient for orient in orientation]

    for i in range(len(body_height)):
        if body_height[i] is None or head_y[i] is None:
            orientation[i] = "Invalid"  # 修改 orientation

    # 遍历每一帧的姿态方向，分割不同的片段
    for i, orient in enumerate(orientation):
        if current_orient is None:
            current_orient, start_frame = orient, i
        elif orient != current_orient:
            end_frame = i - 1
            duration = end_frame - start_frame + 1
            orient_segments.append({
                "orient": current_orient,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_sec": duration / fps,
                "duration_frames": duration
            })
            current_orient, start_frame = orient, i

    # 处理最后一个片段
    if current_orient is not None:
        end_frame = len(orientation) - 1
        duration = end_frame - start_frame + 1
        orient_segments.append({
            "orient": current_orient,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_sec": duration / fps,
            "duration_frames": duration
        })

    return orient_segments

def filter_invalid_orientation_segments(orient_segments, orientation, body_height, head_y, fps=30, min_duration_sec=3, max_duration_sec=90):
    
    min_duration_frames = fps * min_duration_sec
    max_duration_frames = fps * max_duration_sec
    total_frames = len(orientation)
    
    # 计算前 10% 和后 10% 的帧范围
    first_10_percent = min(int(0.1 * total_frames), max_duration_sec)
    last_10_percent = max(int(0.9 * total_frames), total_frames - max_duration_frames)
    
    # 找出所有 "Invalid" 片段
    long_invalid_segments = []
    first_invalid_in_10_percent = None
    last_invalid_in_90_percent = None

    for segment in orient_segments:
        if segment["orient"] == "Invalid":
            # 检查是否超过 min_duration_frames
            if segment["duration_frames"] > min_duration_frames:
                long_invalid_segments.append(segment)

            # 检查是否在前10%
            if segment["end_frame"] < first_10_percent:
                first_invalid_in_10_percent = segment  # 持续更新，找到最后一个
            # 检查是否在后10%
            elif segment["start_frame"] >= last_10_percent and last_invalid_in_90_percent is None:
                last_invalid_in_90_percent = segment  # 仅找到第一个就停

    # 更新 orientation_segments 使得前后片段变为 "Invalid"
    new_orient_segments = []
    invalid_mode = False  # 这个标志决定是否将后续片段设为 "Invalid"

    for segment in orient_segments:
        if first_invalid_in_10_percent and segment["end_frame"] <= first_invalid_in_10_percent["end_frame"]:
            invalid_mode = True  # 触发 Invalid 模式

        if last_invalid_in_90_percent and segment["start_frame"] >= last_invalid_in_90_percent["start_frame"]:
            invalid_mode = True  # 触发 Invalid 模式
        
        if invalid_mode:
            # 将当前片段变为 Invalid
            new_segment = segment.copy()
            new_segment["orient"] = "Invalid"
            new_orient_segments.append(new_segment)
        else:
            new_orient_segments.append(segment)


    # 4️⃣ 删除超过 1 秒的 "Invalid" 片段
    frames_to_remove = set()
    for segment in orient_segments:
        if segment["orient"] == "Invalid":
            frames_to_remove.update(range(segment["start_frame"], segment["end_frame"] + 1))
    
    updated_orient_segments = []
    for segment in orient_segments:
        if segment["orient"] == "Invalid" :
            print(f"Deleted Invalid segment: Start {segment['start_frame']}, End {segment['end_frame']}, Duration {segment['duration_frames']} frames ({segment['duration_sec']} sec)")
        else:
            updated_orient_segments.append(segment)  # 只保留未被删除的片段

    frames_to_keep = set(range(total_frames)) - frames_to_remove
    updated_orientation = [orientation[i] for i in frames_to_keep]
    updated_body_height = [body_height[i] for i in frames_to_keep]
    updated_head_y = [head_y[i] for i in frames_to_keep]
    
    if updated_orient_segments:
        new_segments = []
        prev_end_frame = 0
        
        for seg in updated_orient_segments:
            duration = seg["end_frame"] - seg["start_frame"]
            seg["start_frame"] = prev_end_frame
            seg["end_frame"] = prev_end_frame + duration
            prev_end_frame = seg["end_frame"] + 1
            new_segments.append(seg)
        
        updated_orient_segments = new_segments

    return updated_orient_segments, updated_orientation, updated_body_height, updated_head_y
    
def compute_adaptive_threshold(data, method="std", k=2):
    """
    计算数据的自适应阈值，用于检测异常值。

    该函数根据不同的统计方法计算数据的动态阈值：
    - method="std"  -> 使用标准差计算阈值：threshold = k * std
    - method="mad"  -> 使用平均绝对偏差（MAD）计算阈值
    - method="iqr"  -> 使用 IQR（四分位距）计算阈值

    参数：
        data (list[float]): 需要计算阈值的数据列表，例如 `body_height`。
        method (str): 选择计算方法，可选值为 'std', 'mad', 'iqr' (默认 "std")。
        k (float): 乘法因子，用于控制阈值的灵敏度 (默认 2)。

    返回：
        float: 计算出的自适应阈值。

    异常：
        ValueError: 如果提供的 `method` 不支持，则抛出异常。
    """
    
    # 将数据转换为 NumPy 数组，确保支持数学计算
    data = np.array(data)
    data = np.array([x if x is not None else 0 for x in data])

    if method == "std":
        # 使用标准差计算阈值
        threshold = k * np.std(data)
    
    elif method == "mad":
        # 计算中位数
        median = np.median(data)
        # 计算平均绝对偏差（MAD）
        mad = np.median(np.abs(data - median))
        threshold = k * mad

    elif method == "iqr":
        # 计算第一四分位数（Q1）和第三四分位数（Q3）
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        # 计算四分位距（IQR）
        iqr = Q3 - Q1
        threshold = k * iqr

    else:
        # 如果传入的 `method` 参数不合法，则抛出错误
        raise ValueError("Unsupported method. Choose from ['std', 'mad', 'iqr']")
    
    return threshold

def detect_change_points(data, percentile=95, window_size=3, visualize=False):
    """
    检测时间序列数据中的突变点（突增或突降）。

    主要逻辑：
    1. 计算数据的变化率（取绝对值差分）。
    2. 根据 `percentile` 计算突变阈值。
    3. 使用 `find_peaks` 方法检测超过阈值的突变点。
    4. 过滤掉短暂突变点，确保变化后维持高值一段时间。
    5. 如果 `visualize=True`，则绘制可视化图表。

    参数:
        data (array-like): 输入的时间序列数据 (如 body_height)。
        percentile (float): 变化点的阈值（分位数），默认 95（取前 5% 最大变化点）。
        window_size (int): 用于过滤短暂突变点的窗口大小（默认 3）。
        visualize (bool): 是否可视化检测结果，默认 True。

    返回:
        list[int]: 变化点的索引列表。

    """

    # 确保数据是 numpy 数组
    data = np.array(data, dtype=float)

    # 计算变化率（取绝对值的差分）
    diff = np.abs(np.diff(data))

    # 计算突变点阈值（取前 `percentile%` 最大变化值）
    threshold = np.percentile(diff, percentile)

    # 检测变化点（峰值点）
    jump_points, _ = find_peaks(diff, height=threshold)

    # 过滤掉短暂突变点
    change_points = [
        p for p in jump_points if np.mean(data[p:p + window_size]) > np.mean(data) + np.std(data)
    ]

    # 可视化检测结果
    if visualize:
        plt.figure(figsize=(12, 5))
        plt.plot(data, label="Data", color="blue")
        plt.scatter(change_points, data[change_points], color='red', label="Change Points", zorder=3)
        plt.legend()
        plt.title("Detected Change Points")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.show()

    return change_points

def remove_large_height_changes(change_points, orientation_segments, orientation, body_height, head_y, fps=30, max_duration_sec=90):
    """
    处理 change_points，删除异常突变的片段。

    主要逻辑：
    1. **计算自适应阈值 threshold**：
       - 通过 `compute_adaptive_threshold()` 计算 body_height 的阈值，用于识别异常变化。
    2. **遍历所有 change_points**：
       - 在 `max_duration_frames` 内，寻找最后一个变化点。
       - 计算该区间的 body_height 均值，并与全局均值比较。
    3. **判断是否删除该区间**：
       - 若 body_height 的突变区域与其他区域的均值差值大于 threshold，则删除该片段。
       - 若该突变发生在前 10% 或后 10% 的视频时间，则直接删除对应区域。
    4. **同步更新 orientation_segments、orientation、body_height 和 head_y**：
       - 删除相关帧，并重新计算有效的片段索引。

    参数：
        change_points (list[int]): 检测到的突变点索引。
        orientation_segments (list[dict]): 姿态信息片段。
        orientation (list[str]): 每帧的姿态信息。
        body_height (list[float]): 每帧的身体高度数据。
        head_y (list[float]): 每帧的头部 Y 坐标数据。
        fps (int): 视频帧率，默认 30。
        method (str): 计算 threshold 的方法 ["std", "mad", "iqr"]，默认 "std"。
        k (float): 计算 threshold 时的乘法因子，默认 2。
        max_duration_sec (int): 突变最大持续时间（秒），默认 90。

    返回：
        tuple:
            - updated_orientation_segments (list[dict]): 更新后的姿态片段数据。
            - updated_orientation (list[str]): 过滤后的姿态信息。
            - updated_body_height (list[float]): 过滤后的身体高度数据。
            - updated_head_y (list[float]): 过滤后的头部 Y 轴数据。
    """

    total_frames = len(body_height)
    max_duration_frames = fps * max_duration_sec

    # 计算自适应阈值 threshold
    threshold = compute_adaptive_threshold(body_height)

    frames_to_remove = set()  # 记录需要删除的帧索引

    # 遍历所有突变点，确定需要删除的片段
    for i, cp in enumerate(change_points):
        # 寻找 max_duration_frames 内的最后一个突变点
        future_changes = [p for p in change_points if p > cp and p <= cp + max_duration_frames]

        end_idx = max(future_changes) if future_changes else cp  # 选取最后的突变点
        start_idx = cp  # 记录突变区间起点

        # 计算该区间 body_height 的均值
        region_mean = np.mean(body_height[start_idx:end_idx])

        # 计算全局 body_height 均值（排除该区间）
        other_frames = [i for i in range(total_frames) if i < start_idx or i > end_idx]
        if other_frames:
            other_mean = np.mean([body_height[i] for i in other_frames])
        else:
            other_mean = np.mean(body_height)  # 仅有该区间存在，取全局均值

        # 计算 body_height 的变化幅度
        height_diff = abs(region_mean - other_mean)

        # 处理前 10% 和后 10% 的变化
        has_early_change = any(p <= max_duration_frames for p in [start_idx, end_idx])
        has_late_change = any(p >= total_frames - max_duration_frames for p in [start_idx, end_idx])

        if height_diff > threshold:
            if has_early_change:
                # 删除前 10% 内的突变区域
                frames_to_remove.update(range(0, end_idx + 1))
            elif has_late_change:
                # 删除后 10% 内的突变区域
                frames_to_remove.update(range(start_idx, total_frames))
            else:
                # 删除该突变区间
                frames_to_remove.update(range(start_idx, end_idx + 1))

    # 过滤 orientation_segments，并同步删除相应帧的数据
    new_frames_to_remove = frames_to_remove.copy()

    updated_orientation_segments = []
    for seg in orientation_segments:
        # 如果该片段的帧被标记为删除，则丢弃
        if not any(frame in frames_to_remove for frame in range(seg["start_frame"], seg["end_frame"] + 1)):
            updated_orientation_segments.append(seg)
        else:
            new_frames_to_remove.update(range(seg["start_frame"], seg["end_frame"] + 1))

    # 重新计算有效帧索引
    frames_to_keep = set(range(total_frames)) - new_frames_to_remove
    updated_orientation = [orientation[i] for i in frames_to_keep]
    updated_body_height = [body_height[i] for i in frames_to_keep]
    updated_head_y = [head_y[i] for i in frames_to_keep]

    # 重新计算 orientation_segments，使索引保持连续
    if updated_orientation_segments:
        new_segments = []
        prev_end_frame = 0

        for seg in updated_orientation_segments:
            duration = seg["end_frame"] - seg["start_frame"]
            seg["start_frame"] = prev_end_frame
            seg["end_frame"] = prev_end_frame + duration
            prev_end_frame = seg["end_frame"] + 1
            new_segments.append(seg)

        updated_orientation_segments = new_segments

    return updated_orientation_segments, updated_orientation, updated_body_height, updated_head_y

def merge_alternating_orients(orientation_segments, fps=30, max_swaps=15, min_duration_sec=3):
    """
    合并短时交替的姿态片段，以减少抖动和误判。

    主要逻辑：
    1. **合并相邻的相同姿势片段**：
       - 如果当前片段的方向与上一个相同，则合并它们，更新 `end_frame` 和 `duration`。
    2. **遍历姿态片段，合并交替变换的短时片段**：
       - 记录 `swap_count`，跟踪短时交替出现的次数。
       - 如果 `swap_count` 超过 `max_swaps`，则合并该区域，并创建一个新的片段。
       - 否则，保持原状，防止过度合并。
    3. **处理未被合并的片段**：
       - 追加最后一个未处理的片段，防止数据丢失。

    参数：
        orientation_segments (list[dict]): 姿态信息片段，每个片段是一个字典。
        fps (int): 视频的帧率（默认 30）。
        max_swaps (int): 允许的最大交替切换次数，超过此值则进行合并（默认 15）。
        min_duration_sec (int): 最小持续时间（秒），小于该值的片段将被合并（默认 3 秒）。

    返回：
        list[dict]: 处理后的 `orientation_segments` 片段列表。
    """

    # **合并相邻的相同姿势片段**
    merged_segments = []
    for segment in orientation_segments:
        if merged_segments and merged_segments[-1]["orient"] == segment["orient"]:
            # 如果当前片段与上一个片段方向相同，则合并
            merged_segments[-1]["end_frame"] = segment["end_frame"]
            merged_segments[-1]["duration_sec"] = (merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1) / fps
            merged_segments[-1]["duration_frames"] = merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1
        else:
            merged_segments.append(segment)
    
    # 更新 `orientation_segments`
    orientation_segments = merged_segments 

    # 计算最小持续时间（转换为帧数）
    min_duration_frames = fps * min_duration_sec
    result = []
    i = 0  # 迭代索引
    
    while i < len(orientation_segments) - 1:
        current_orient = orientation_segments[i]['orient']
        current_frame = orientation_segments[i]['duration_frames']
        if current_frame < min_duration_frames:
            swap_count = 0  # 记录交替变换次数
            combined_segments = [orientation_segments[i]]  # 存储待合并片段
            next_orient = orientation_segments[i + 1]['orient']
            next_frame = orientation_segments[i + 1]['duration_frames']
            j = i + 1  # 用于收集后续片段的索引
            
            # **如果当前片段时长较短，且下一个片段的方向不同，则尝试合并**
            if current_orient != next_orient and next_frame < min_duration_frames:
                combined_segments.append(orientation_segments[j])
                j += 1
                
                # **继续查找更多的短时交替片段**
                while j < len(orientation_segments):
                    third_orient = orientation_segments[j]['orient']
                    third_segment = orientation_segments[j]
                    
                    # **如果第三个片段的方向属于 (current_orient, next_orient)，且短时交替，则继续合并**
                    if (third_orient in [current_orient, next_orient] and
                        third_orient != combined_segments[-1]['orient'] and
                        third_segment['duration_frames'] < min_duration_frames):
                        swap_count += 1  # 记录交替切换次数
                        combined_segments.append(third_segment)
                        j += 1  # 继续遍历
                    else:
                        break  # 规则被破坏，停止合并
                
                # **如果交替切换次数超过 `max_swaps`，合并这些片段**
                if swap_count > max_swaps:
                    combined_orient = f"{current_orient}-{next_orient}"  # 组合方向
                    merged_segment = {
                        'orient': combined_orient,
                        'start_frame': combined_segments[0]['start_frame'],
                        'end_frame': combined_segments[-1]['end_frame'],
                        'duration_sec': sum(seg['duration_sec'] for seg in combined_segments),
                        'duration_frames': sum(seg['duration_frames'] for seg in combined_segments)
                    }
                    result.append(merged_segment)  # 存储合并后的片段
                    print(merged_segment)  # 打印合并信息（可选）
                else:
                    result.extend(combined_segments)  # 交替次数较少，不合并
                
                # **跳到下一个未处理的片段**
                i = j  
            else:
                # **当前片段不符合合并条件，直接添加到结果**
                result.append(orientation_segments[i])
                i += 1  # 继续主循环遍历
        else:
            # **当前片段不符合合并条件，直接添加到结果**
                result.append(orientation_segments[i])
                i += 1  # 继续主循环遍历
    
    # **追加最后一个 segment，如果它未被处理**
    if i == len(orientation_segments) - 1:
        result.append(orientation_segments[i])
    
    return result

def merge_orientation_segments(orientation_segments, orientation, body_height, head_y, fps=30, min_duration_sec=3, max_duration_sec=15):
    """
    合并短时的姿态片段，去除不稳定的片段，并优化方向数据。

    主要逻辑：
    1. **合并短片段**：
       - 如果片段 `duration_frames < min_duration_frames`，则合并到前一个姿势段，直到所有短片段被合并完毕。
    2. **合并相邻的相同姿势片段**：
       - 如果相邻片段的 `orient` 相同，则合并。
    3. **移除时长小于 max_duration_sec 的首尾片段**：
       - 如果首尾片段的 `duration_frames < max_duration_frames`，则删除该片段的帧数据。
    4. **调整短片段的姿势**：
       - 如果片段 `duration_frames < max_duration_frames`，并且它的前后片段方向相同，则设为该方向，否则设为前一个方向。
    5. **再次合并相邻的相同姿势片段**：
       - 避免因调整姿势后产生的重复片段。

    参数：
        orientation_segments (list[dict]): 姿态信息片段，每个片段是一个字典。
        orientation (list[str]): 每帧的姿态信息。
        body_height (list[float]): 每帧的身体高度数据。
        head_y (list[float]): 每帧的头部 Y 轴数据。
        fps (int): 视频的帧率（默认 30）。
        min_duration_sec (int): 最小持续时间（秒），小于该值的片段将被合并（默认 3 秒）。
        max_duration_sec (int): 最大合并时长（秒），超过该值的片段才会被保留（默认 15 秒）。

    返回：
        tuple:
            - orientation_segments (list[dict]): 处理后的姿态片段。
            - orientation (list[str]): 过滤后的姿态信息。
            - body_height (list[float]): 过滤后的身体高度数据。
            - head_y (list[float]): 过滤后的头部 Y 轴数据。
    """

    # **计算帧数阈值**
    min_duration_frames = fps * min_duration_sec
    max_duration_frames = fps * max_duration_sec

    # **第一步：合并短片段，直到所有短片段被合并完毕**
    final_segments = orientation_segments[:]
    while True:
        updated_segments = []
        merged = False  # 记录是否发生了合并

        for i, segment in enumerate(final_segments):
            if updated_segments and segment["duration_frames"] < min_duration_frames:
                # **将短片段合并到前一个姿势段**
                updated_segments[-1]["end_frame"] = segment["end_frame"]
                updated_segments[-1]["duration_sec"] = (
                    updated_segments[-1]["end_frame"] - updated_segments[-1]["start_frame"] + 1
                ) / fps
                updated_segments[-1]["duration_frames"] = (
                    updated_segments[-1]["end_frame"] - updated_segments[-1]["start_frame"] + 1
                )
                merged = True  # 记录合并发生
            else:
                updated_segments.append(segment)

        if not merged:
            break  # 没有发生合并，跳出循环
        
        final_segments = updated_segments
        orientation_segments = final_segments  # 更新 segments

    # **第二步：合并相邻的相同姿势片段**
    merged_segments = []
    for segment in orientation_segments:
        if merged_segments and merged_segments[-1]["orient"] == segment["orient"]:
            # **合并相邻相同姿势片段**
            merged_segments[-1]["end_frame"] = segment["end_frame"]
            merged_segments[-1]["duration_sec"] = (merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1) / fps
            merged_segments[-1]["duration_frames"] = merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1
        else:
            merged_segments.append(segment)
    
    orientation_segments = merged_segments  # 更新 segments

    frames_to_remove = set()  # 记录要删除的帧

     # 从头开始遍历
    while orientation_segments:
        first_segment = orientation_segments[0]
        if first_segment['duration_frames'] < max_duration_frames:
            print(f"🗑 删除头部片段 (小于 {max_duration_sec} 秒): {first_segment}")
            frames_to_remove.update(range(first_segment['start_frame'], first_segment['end_frame'] + 1))
            orientation_segments.pop(0)
        else:
            break  # 遇到符合要求的片段，停止从头部遍历

    # 从尾部开始遍历
    while orientation_segments:
        last_segment = orientation_segments[-1]
        if last_segment['duration_frames'] < max_duration_frames:
            print(f"🗑 删除尾部片段 (小于 {max_duration_sec} 秒): {last_segment}")
            frames_to_remove.update(range(last_segment['start_frame'], last_segment['end_frame'] + 1))
            orientation_segments.pop(-1)
        else:
            break  # 遇到符合要求的片段，停止从尾部遍历

     # **第四步：删除最后一个小于 max_duration_sec 的片段**
    if orientation_segments and orientation_segments[-1]["duration_frames"] < 2 * max_duration_frames and orientation_segments[-1]['orient'] == 'neutral':
        last_segment = orientation_segments[-1]
        print(f"🗑 删除尾部片段 (小于 {max_duration_sec} 秒): {last_segment}")
        frames_to_remove.update(range(last_segment["start_frame"], last_segment["end_frame"] + 1))
        orientation_segments.pop(-1)  # 删除片段

    # **第五步：删除 `orientation`、`body_height` 和 `head_y` 中的相应帧**
    orientation = [orient for i, orient in enumerate(orientation) if i not in frames_to_remove]
    body_height = [height for i, height in enumerate(body_height) if i not in frames_to_remove]
    head_y = [head_y for i, head_y in enumerate(head_y) if i not in frames_to_remove]

    # **第六步：重新调整 segment 索引**
    if orientation_segments:
        new_segments = []
        prev_end_frame = 0
        
        for seg in orientation_segments:
            duration = seg["end_frame"] - seg["start_frame"]
            seg["start_frame"] = prev_end_frame
            seg["end_frame"] = prev_end_frame + duration
            prev_end_frame = seg["end_frame"] + 1
            new_segments.append(seg)
        
        orientation_segments = new_segments  # 更新 segments

    # **第七步：调整短片段的方向**
    for i in range(1, len(orientation_segments) - 1):  # 避免访问超出范围
        segment = orientation_segments[i]

        if segment["duration_frames"] < max_duration_frames:
            prev_orient = orientation_segments[i - 1]["orient"]

            # **如果前后姿势相同，则设为该姿势，否则设为前一个片段的姿势**
            segment["orient"] = prev_orient

    # **第八步：再次合并相邻相同姿势**
    merged_segments = []
    for segment in orientation_segments:
        if merged_segments and merged_segments[-1]["orient"] == segment["orient"]:
            merged_segments[-1]["end_frame"] = segment["end_frame"]
            merged_segments[-1]["duration_sec"] = (merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1) / fps
            merged_segments[-1]["duration_frames"] = merged_segments[-1]["end_frame"] - merged_segments[-1]["start_frame"] + 1
        else:
            merged_segments.append(segment)

    orientation_segments = merged_segments  # 更新 segments
    
    return orientation_segments, orientation, body_height, head_y

def split_head_y_by_orientation(orientation_segments, head_y):
    """
    根据 orientation_segments 中的 start_frame 和 end_frame，分割 head_y 数据。

    主要逻辑：
    1. **遍历 orientation_segments**：
       - 每个片段包含 `start_frame` 和 `end_frame`，用于确定数据分割范围。
    2. **提取 head_y 片段**：
       - 取 `head_y[start:end+1]`，确保 `end_frame` 对应的帧也被包含在内。
    3. **存储分割后的 head_y 片段**：
       - 将切片结果存入 `segmented_head_y` 列表中，保持索引一致。

    参数：
        orientation_segments (list[dict]): 姿态信息片段，每个片段包含 `start_frame` 和 `end_frame`。
        head_y (list[float]): 头部 Y 坐标数据列表。

    返回：
        list[list[float]]: 分割后的 head_y 片段列表，每个片段对应一个 `orientation_segments` 片段。
    """

    segmented_head_y = []  # 存储分割后的 head_y 片段
    
    for segment in orientation_segments:
        start = segment['start_frame']
        end = segment['end_frame'] + 1  # 包含 `end_frame` 所在的索引
        head_y_segment = head_y[start:end]  # 提取对应的 head_y 数据
        
        segmented_head_y.append(head_y_segment)

    return segmented_head_y

def process_segmented_head_y(segmented_head_y, frame_window=400, max_timestamps=8, smooth_window=5, max_iterations=10):
    """
    处理 segmented_head_y，迭代检测突变点，分割数据，清理无效数据，并平滑断点。

    主要逻辑：
    1. **迭代处理数据**（最多 `max_iterations` 次）：
       - 逐个检查 `segmented_head_y`，移除短片段，并检测突变点。
    2. **检测并标记突变点**：
       - 计算 `threshold` 作为变化检测标准。
       - 使用 `ruptures` 进行突变点检测，识别显著变化区域。
    3. **去除无效数据**：
       - 若突变点在前 `frame_window` 帧或后 `frame_window` 帧，则标记为无效。
       - 对相邻突变点进行合并，减少误判。
       - 若突变点数量超过 `max_timestamps`，跳过该段数据，防止误分割。
    4. **分割数据**：
       - 依据突变点对数据进行分割，避免数据混乱。
       - 若相邻突变点间距过短，则跳过分割，以避免碎片化。
    5. **平滑数据**：
       - 对于每个分割片段，使用 `savgol_filter` 进行平滑，以减少噪声。
    6. **终止条件**：
       - 若数据在某次迭代后不再发生变化，则终止迭代，避免无限循环。

    参数：
        segmented_head_y (list of list): 头部 Y 轴数据，每个子列表表示一个时间序列片段。
        frame_window (int): 用于检测突变点的前后窗口大小（默认 400 帧）。
        max_timestamps (int): 允许的最大突变点数量，超出则跳过该段数据（默认 8）。
        smooth_window (int): 平滑窗口大小（默认 5）。
        max_iterations (int): 最大迭代次数，防止无限循环（默认 10）。

    返回：
        tuple:
            - processed_data (list of list): 处理后的分割数据。
            - split_info (list): 记录 `segmented_head_y` 的第几个元素被分割几次。
    """

    # **初始输入**
    processed_data = segmented_head_y
    split_info = list(range(len(segmented_head_y)))  # 记录初始索引
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        new_processed_data = []
        new_split_info = []

        has_changes = False  # 追踪是否有新的分割或数据清理

        for idx, segment in zip(split_info, processed_data):
            if len(segment) < 2 * frame_window:  # **数据过短，则跳过**
                new_processed_data.append(segment)
                new_split_info.append(idx)
                continue

            segment = np.array(segment, dtype=float)

            # ✅ **1. 计算自适应阈值**
            threshold = compute_adaptive_threshold(segment, "std", 1)

            # ✅ **2. 检测突变点**
            algo = rpt.Pelt(model="l2").fit(segment)
            change_points = algo.predict(pen=1)  # 获取突变点索引

            # ✅ **3. 处理突变点**
            invalid_indices = set()
            timestamps = []
            middle_timestamps = []  # 记录中间突变点（排除前 400 帧和后 400 帧）

            for cp in change_points:
                if cp < frame_window:  # **前 400 帧内**
                    before_mean = np.mean(segment[:cp])
                    after_mean = np.mean(segment[cp:cp + frame_window])
                    if abs(after_mean - before_mean) > threshold:
                        invalid_indices.update(range(0, cp + frame_window))

                elif cp > len(segment) - frame_window:  # **后 400 帧内**
                    before_mean = np.mean(segment[cp - frame_window:cp])
                    after_mean = np.mean(segment[cp:])
                    if abs(after_mean - before_mean) > threshold:
                        invalid_indices.update(range(cp - frame_window, len(segment)))

                else:  # **中间部分**
                    before_mean = np.mean(segment[cp - frame_window:cp])
                    after_mean = np.mean(segment[cp:cp + frame_window])
                    if abs(after_mean - before_mean) > threshold:
                        timestamps.append(cp)
                        middle_timestamps.append(cp)  # 记录中间部分突变点

            # ✅ **4. 处理中间的突变点**
            if len(middle_timestamps) > max_timestamps:
                print(f"Skipping segment {idx} due to too many middle timestamps: {len(middle_timestamps)}")
                timestamps = []  # 清空突变点，避免误分割

            # ✅ **5. 处理相邻突变点**
            new_timestamps = []
            last_cp = None

            for cp in timestamps:
                if last_cp is not None and (cp - last_cp) < frame_window:
                    invalid_indices.update(range(last_cp, cp))  # 标记该数据无效
                else:
                    new_timestamps.append(cp)
                    last_cp = cp
            timestamps = new_timestamps  # **更新 timestamps**

            # ✅ **6. 去除无效数据**
            valid_indices = sorted(set(range(len(segment))) - invalid_indices)
            filtered_segment = segment[valid_indices]  # 仅保留有效数据

            if len(valid_indices) < len(segment):  # **数据被修改**
                has_changes = True

            # ✅ **7. 分割数据**
            split_segments = []
            last_cp = 0
            for cp in timestamps:
                if cp - last_cp > 10:  # **避免分割太短**
                    split_segments.append(filtered_segment[last_cp:cp])
                    new_split_info.append(idx)
                last_cp = cp

            if last_cp < len(filtered_segment):  # **添加最后一个片段**
                split_segments.append(filtered_segment[last_cp:])
                new_split_info.append(idx)

            if len(split_segments) > 1:  # **发生了分割**
                has_changes = True

            # ✅ **8. 平滑断点**
            final_segments = []
            for sub_segment in split_segments:
                if len(sub_segment) > smooth_window:
                    sub_segment = savgol_filter(sub_segment, smooth_window, polyorder=2)
                final_segments.append(sub_segment)

            new_processed_data.extend(final_segments)

        # **检查是否还有变化**
        if not has_changes:
            print(f"Converged after {iteration} iterations.")
            break

        # **更新 processed_data 和 split_info**
        processed_data = new_processed_data
        split_info = new_split_info

    return processed_data, split_info

def detect_periodicity_acf_with_peaks(data, threshold=0.2, max_lag=300, min_ratio=0.4, min_alternations=6):
    """
    使用自相关函数 (ACF) 检测时间序列是否具有周期性，并计算最高峰值和最低峰值。

    主要逻辑：
    1. **计算 ACF（自相关函数）**：
       - 计算 `max_lag` 内的自相关值，用于分析数据的周期性。
    2. **统计滞后步长中显著相关的比例**：
       - 计算 `|ACF| > threshold` 的滞后值占比 `ratio`。
    3. **计算 ACF 的符号变化**：
       - 计算 `sign_changes`（ACF 的正负号）。
       - 计算 `alternation_count`（ACF 的正负交替次数）。
    4. **判断周期性**：
       - 只有当 `ratio > min_ratio` 且 `alternation_count >= min_alternations` 时，认为数据具有周期性。
    5. **计算均值和振幅**：
       - 计算数据的 `mean`。
       - 计算 `amp`（数据的 FFT 振幅，需调用 `compute_amplitude_fft`）。

    参数：
        data (array-like): 时间序列数据。
        threshold (float): 自相关系数阈值，绝对值大于此值才算显著相关（默认 0.2）。
        max_lag (int): 计算 ACF 时的最大滞后步长（默认 300）。
        min_ratio (float): 多少比例的滞后值需要超过 `threshold` 才算周期性（默认 0.4）。
        min_alternations (int): 至少多少次正负交替才算周期性（默认 6）。

    返回：
        tuple:
            - periodic (bool): 是否存在周期性。
            - mean (float): 数据均值。
            - amp (float): 数据的 FFT 振幅。
    """

    # **计算 ACF**
    acf_values = sm.tsa.acf(data, nlags=max_lag)

    # **统计 |ACF| 超过 threshold 的滞后点数量**
    above_threshold = np.sum(np.abs(acf_values[1:]) > threshold)  # 统计显著相关的点
    ratio = above_threshold / max_lag  # 计算占比

    # **计算 ACF 的正负变化**
    sign_changes = np.sign(acf_values[1:])  # 获取 ACF 的正负号 (+1 或 -1)
    alternation_count = np.sum(np.diff(sign_changes) != 0)  # 计算正负交替次数

    # **判断周期性**
    periodic = (ratio > min_ratio) and (alternation_count >= min_alternations)

    # **计算均值**
    mean = np.mean(data)

    # **计算数据的 FFT 振幅**
    amp = compute_amplitude_fft(data)  # 需要 `compute_amplitude_fft()` 方法支持

    return periodic, mean, amp

def split_orientation_segments(orientation_segments, segmented_head_y, split_info):
    """
    根据 segmented_head_y 和 split_info 对 orientation_segments 进行相应的分割，
    并按比例分配帧数，以保持数据完整性。

    主要逻辑：
    1. **计算原始片段的 frame 分配情况**：
       - 记录每个 `orientation_segments` 片段对应的 `segmented_head_y` 片段总长度 (`segment_lengths`)。
    2. **遍历 segmented_head_y 并进行分割**：
       - 按比例计算新的 `duration_frames`，确保帧数分配合理。
       - 确保 `start_frame` 和 `end_frame` 连续，避免数据不连贯。
    3. **生成新片段**：
       - 计算新的 `duration_sec`，保持 `FPS` 一致。
       - 创建新的 `orientation_segments`，存储在 `new_segments` 列表中。

    参数：
        orientation_segments (list[dict]): 原始姿态片段，每个字典包含:
            - "orient": 姿态方向
            - "start_frame": 片段起始帧索引
            - "end_frame": 片段结束帧索引
            - "duration_sec": 片段持续时间（秒）
            - "duration_frames": 片段持续帧数
        segmented_head_y (list[list[float]]): 分割后的时间序列数据，每个子列表对应一个分割部分。
        split_info (list[int]): 指示 `segmented_head_y` 每个元素属于哪个 `orientation_segments` 片段。

    返回：
        list[dict]: 重新分割后的 `orientation_segments` 片段列表。
    """

    new_segments = []

    # 记录每个原始片段的 frame 分配情况
    segment_allocations = {}  

    # 计算每个 segment_index 关联的 segmented_head_y 片段总长度
    segment_lengths = {}
    for i, segment_data in enumerate(segmented_head_y):
        segment_index = split_info[i]
        segment_lengths[segment_index] = segment_lengths.get(segment_index, 0) + len(segment_data)

    # 遍历 segmented_head_y 并进行分割
    for i, segment_data in enumerate(segmented_head_y):
        segment_index = split_info[i]  # 该数据片段属于哪个原 `orientation_segments` 片段
        orig_segment = orientation_segments[segment_index]  # 获取原始 `orientation_segments` 片段

        # 获取原始片段的信息
        orig_start_frame = orig_segment["start_frame"]
        orig_duration_frames = orig_segment["duration_frames"]

        # 按比例计算新的 `duration_frames`
        total_segment_length = segment_lengths[segment_index]  # 该片段所有 `segmented_head_y` 数据总长度
        ratio = len(segment_data) / total_segment_length
        new_duration_frames = round(ratio * orig_duration_frames)

        # 确保片段是连续的
        if segment_index not in segment_allocations:
            segment_allocations[segment_index] = orig_start_frame
        start_frame = segment_allocations[segment_index]
        end_frame = start_frame + new_duration_frames - 1

        # 计算帧率 (FPS) 以转换 `duration_frames -> duration_sec`
        fps = orig_segment["duration_sec"] / orig_duration_frames
        duration_sec = new_duration_frames * fps

        # 生成新片段
        new_segment = {
            "orient": orig_segment["orient"],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_sec": duration_sec,
            "duration_frames": new_duration_frames,
        }
        new_segments.append(new_segment)

        # 更新起始位置，确保下一片段的 `start_frame` 连续
        segment_allocations[segment_index] = end_frame + 1

    return new_segments

def compute_amplitude_fft(time_series):
    """
    计算主频及其对应的振幅（基于 FFT）。

    主要逻辑：
    1. **计算 FFT（快速傅里叶变换）**：
       - 获取 `time_series` 的频谱信息。
    2. **计算振幅谱**：
       - 归一化计算振幅，使得振幅大小独立于数据长度。
    3. **获取主频的振幅**：
       - 仅使用正频率部分（FFT 结果的前半部分）。
       - 忽略零频（直流分量），找到振幅最大的频率分量。

    参数：
        time_series (array-like): 输入的时间序列数据。

    返回：
        float: 主频对应的振幅。
    """

    N = len(time_series)  # **数据长度**
    fft_values = np.fft.fft(time_series)  # **计算 FFT**
    
    # **计算振幅谱（归一化处理）**
    amplitude_spectrum = (2 / N) * np.abs(fft_values)  # **振幅归一化**

    # **取正频率部分（去掉负频率）**
    positive_amplitude = amplitude_spectrum[:N // 2]

    # **找到主频索引（忽略零频）**
    main_freq_index = np.argmax(positive_amplitude[1:]) + 1  # **跳过直流分量（DC）**
    main_amplitude = positive_amplitude[main_freq_index]

    return main_amplitude

def update_orientation_segments(orientation_segments, periodics, means, amps):
    """
    根据 `periodics`、`means` 和 `amps` 更新 `orientation_segments`，添加 `head_y` 值：
    
    主要逻辑：
    1. **判断是否存在周期性**：
       - 若 `periodics[i] == True`，则 `head_y = [means[i] - amps[i], means[i] + amps[i]]`。
       - 若 `periodics[i] == False`，则 `head_y = means[i]`（无明显周期性，直接赋值）。
    2. **更新 `orientation_segments`**：
       - 遍历 `orientation_segments`，为每个片段计算 `head_y` 并存入字典。

    参数：
        orientation_segments (list[dict]): 待更新的姿态片段列表，每个字典包含：
            - "orient": 姿态方向
            - "start_frame": 片段起始帧索引
            - "end_frame": 片段结束帧索引
            - "duration_sec": 片段持续时间（秒）
            - "duration_frames": 片段持续帧数
        periodics (list[bool]): 是否存在周期性 (True / False)。
        means (list[float]): 每个片段的均值。
        amps (list[float]): 每个片段的振幅（周期性振幅）。

    返回：
        list[dict]: 包含 `head_y` 信息的 `orientation_segments`。
    """

    for i in range(len(orientation_segments)):
        if periodics[i]:
            orientation_segments[i]["head_y"] = [means[i] - amps[i], means[i] + amps[i]]  # **设定区间**
        else:
            orientation_segments[i]["head_y"] = means[i]  # **无周期性，直接赋值**

    return orientation_segments

def plot_orientation_segments(orientation_segments, save_path):
    """
    绘制 `head_y` 变化（基于 `orientation_segments["head_y"]`）并填充片段下方的区域，
    处理片段间断点，并在 `Static` 片段上覆盖交叉线，同时标注 `orient` 方向。

    参数：
        orientation_segments (list[dict]): 姿态片段列表，每个字典包含：
            - "start_frame": 片段起始帧索引。
            - "end_frame": 片段结束帧索引。
            - "head_y": 头部高度 (单值或 `[min, max]` 区间)。
            - "orient": 姿势方向（如 "neutral", "right", "up", "down"）。

    返回：
        None: 直接在 `matplotlib` 画布上绘制图像，不返回值。
    """

    if not orientation_segments:
        print("错误: orientation_segments 为空，无法绘制图表。")
        return
    
    try:
        # 读取图片
        # 读取图片
        img_path = "full_body.png"
        img = Image.open(img_path)
        img_width, img_height = img.size
        aspect_ratio = img_width / img_height
        print(f"✅ 图片加载成功: {img_path} (宽: {img_width}, 高: {img_height})")
    except Exception as e:
        print(f"❌ 图片加载失败: {e}")
        return
    
    # 创建主图
    target_width_px = 1900  # 宽度 1920 像素
    target_height_px = 700  # 高度 1080 像素
    dpi = 100  # 每英寸的像素点数
    # 转换为英寸
    fig_width = target_width_px / dpi
    fig_height = target_height_px / dpi
    # 创建图形
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # **定义颜色映射**
    color_map = {
        'neutral': '#8dd3c7',
        # 'right': '#ffffb3',
        'up': '#fb8072',
        'down': '#bebada',
        'down-neutral': '#fdb462',
        'neutral-down': '#fdb462',
        'up-neutral': '#b3de69',
        'neutral-up': '#b3de69',
        'down-up': '#fccde5',
        'up-down': '#fccde5'
    }

    # **遍历 orientation_segments，绘制 head_y 轨迹**
    for entry in orientation_segments:
        start_time = entry["start_frame"]
        end_time = entry["end_frame"]
        head_y = entry["head_y"]
        orient = entry["orient"]

        # **获取颜色**
        color = color_map.get(orient, 'gray')

        # **生成 x 轴数据**
        x_values = np.arange(start_time, end_time + 1)

        # **生成 y 轴数据**
        if isinstance(head_y, (int, float)):  # **单值，绘制水平直线**
            y_values = np.full_like(x_values, head_y, dtype=float)

        elif isinstance(head_y, (list, tuple)) and len(head_y) == 2:  # **区间值，绘制振荡曲线**
            min_val, max_val = head_y
            num_points = len(x_values)
            num_oscillations = 3  # 指定往返的次数

            # 中间值 (起点与终点)
            mid_val = (min_val + max_val) / 2

            # 计算每次往返占用的点数（两个来回为一组）
            points_per_oscillation = num_points // (num_oscillations * 4)
            
            indices = []
            for _ in range(num_oscillations):
                # 生成一个完整的往返：中间值 -> max_val -> 中间值 -> min_val -> 中间值
                indices.extend(np.linspace(mid_val, max_val, points_per_oscillation))   # 中间值 -> 最大值
                indices.extend(np.linspace(max_val, mid_val, points_per_oscillation))   # 最大值 -> 中间值
                indices.extend(np.linspace(mid_val, min_val, points_per_oscillation))   # 中间值 -> 最小值
                indices.extend(np.linspace(min_val, mid_val, points_per_oscillation))   # 最小值 -> 中间值

            # 如果点数不够，补上中间点
            if len(indices) < num_points:
                indices = np.concatenate([indices, [mid_val] * (num_points - len(indices))])
            
            y_values = np.array(indices[:num_points])  # 确保 y_values 的长度与 x_values 一致
                
        else:
            continue  # **数据格式错误，跳过**

        # **填充曲线下方的区域**
        plt.fill_between(x_values, y_values, 0, color=color, alpha=0.5, label=orient if orient not in plt.gca().get_legend_handles_labels()[1] else "")

        # **在 orientation 片段顶部标注 orient**
        mid_x = (start_time + end_time) / 2
        mid_y = max(y_values) + 0.01  # **让文本稍微高于曲线**
        if '-' in orient:  # 如果是连接词
            word1, word2 = orient.split('-')
            # 判断较长的单词和较短的单词
            if len(word1) >= len(word2):
                plt.text(mid_x, mid_y + 0.03, word1, fontsize=10 , ha='center', va='bottom', color='black', fontfamily='Arial')
                plt.text(mid_x, mid_y, f'&{word2}', fontsize=10, ha='center', va='bottom', color='black', fontfamily='Arial')
            else:
                plt.text(mid_x, mid_y + 0.03, word2, fontsize=10, ha='center', va='bottom', color='black', fontfamily='Arial')
                plt.text(mid_x, mid_y, f'&{word1}', fontsize=10, ha='center', va='bottom', color='black', fontfamily='Arial')
        else:  # 如果是单词
            plt.text(mid_x, mid_y, orient, fontsize=10, ha='center', va='bottom', color='black', fontfamily='Arial')


    # **添加图例、标签、网格**
    plt.ylim(0, 1.1)
    plt.xlabel("Frame Index")
    plt.ylabel("Nose Height (Normalized)")
    plt.title("Nose Height and Facial Orientation Over Time")
    plt.legend(prop={'family': 'Arial'})
    plt.grid(True, linestyle='--', alpha=0.6)

    # 在左侧添加图片
    target_height =  0.72
    target_width = target_height * aspect_ratio
    ax_img = fig.add_axes([0.03, 0.1, target_width, target_height], anchor='W')  # 确保图片的高度与 0-1 对齐
    ax_img.imshow(img)
    ax_img.axis('off')
    ax_img.set_zorder(0)

    # 保存图像到指定路径
    plt.savefig(save_path)
    plt.close(fig)












app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

video_filename = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/overviews")
def overviews():
    return render_template("overviews.html")

@app.route("/upload")
def upload():
    return render_template("index.html")  # 这里可以用 index.html

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/upload", methods=["POST"])
def handle_upload():
    global video_filename
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded."})

    file = request.files["video"]
    video_filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_filename)

    return jsonify({"status": "success"})

@app.route("/get_video")
def get_video():
    return jsonify({"video_url": "/" + video_filename if video_filename else None})

@app.route("/check_status")
def check_status():
    frame_data_list, fps= generate_video(video_filename)
    body_height, orientation, head_y = extract_data_from_frame_list(frame_data_list)
    orientation = smooth_stable_data(orientation)
    orientation_segments = first_orientation_segments(orientation, body_height, head_y, fps)
    orientation_segments, orientation, body_height, head_y = filter_invalid_orientation_segments(orientation_segments, orientation, body_height, head_y, fps)
    
    change_points = detect_change_points(body_height, visualize=False)
    orientation_segments, orientation, body_height, head_y = remove_large_height_changes(change_points, orientation_segments, orientation, body_height, head_y, fps)
    orientation_segments = merge_alternating_orients(orientation_segments, fps)
    orientation_segments, orientation, body_height, head_y = merge_orientation_segments(orientation_segments, orientation, body_height, head_y, fps)


    segmented_head_y = split_head_y_by_orientation(orientation_segments, head_y)
    segmented_head_y, split_info = process_segmented_head_y(segmented_head_y)
    print(split_info)

    periodics = []
    means = []
    amps = []
    for segment in segmented_head_y:
        segment = np.array(segment, dtype=float)
        periodic, mean, amp = detect_periodicity_acf_with_peaks(segment)
        if periodic:
            if amp < 0.05:
                periodic = False
        periodics.append(periodic)
        means.append(mean)
        amps.append(amp)
    
    orientation_segments = split_orientation_segments(orientation_segments, segmented_head_y, split_info)
    print(periodics)
    orientation_segments = update_orientation_segments(orientation_segments, periodics, means, amps)
    
    image_path = os.path.join(UPLOAD_FOLDER, 'result_plot.png')
    plot_orientation_segments(orientation_segments, image_path)


    segment1 = "This is a short sentence."
    segment2 = "This is a slightly longer sentence to test different lengths."
    segment3 = "This is a medium-length sentence designed to provide a more realistic placeholder."
    segment4 = "This is a very long sentence meant to serve as a placeholder for testing how longer text appears when inserted into the summary template, ensuring that the formatting remains intact and readable."
    segment5 = "Another short sentence to complete the set."

    summary_template = """Video analysis complete! The video contains several interesting segments:\n
    - 1. {Segment1}\n
    - 2. {Segment2}\n
    - 3. {Segment3}\n
    - 4. {Segment4}\n
    - 5. {Segment5}"""

    # 用 format() 方法替换占位符
    analysis_result = {
        "summary": summary_template.format(
            Segment1=segment1,
            Segment2=segment2,
            Segment3=segment3,
            Segment4=segment4,
            Segment5=segment5
        ),
        "image_url": "/" + image_path if image_path else None
    }
    
    return jsonify({
            "done": True,
            "result": analysis_result["summary"],
            "image_url": analysis_result["image_url"]
        })

# if __name__ == "__main__":
#     app.run(debug=True)


