
from flask import Flask, request, jsonify, render_template

import os, cv2, shutil, time

import mediapipe as mp

import matplotlib.pyplot as plt

from tqdm import tqdm


app = Flask(__name__)


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
pose = mp_pose.Pose(static_image_mode=False,        # 是静态图片还是连续视频帧
                    model_complexity=2,            # 选择人体姿态关键点检测模型，0性能差但快，2性能好但慢，1介于两者之间
                    smooth_landmarks=True,         # 是否平滑关键点
                    min_detection_confidence=0.5,  # 置信度阈值
                    min_tracking_confidence=0.5)   # 追踪阈值


# 确保上传文件的文件夹存在
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def home():
    return render_template('index.html')


def process_frame(img):

    rating = 0
    
    # 获取图像宽高
    h, w = img.shape[0], img.shape[1]

    # BGR转RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型，获取预测结果
    results = pose.process(img_RGB)

    if results.pose_landmarks: # 若检测出人体关键点

        # 可视化关键点及骨架连线
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cx_list = []
        cy_list = []
        cz_list = []

        scaler = 1

        for i in range(33):  # 遍历所有33个关键点
            # 获取该关键点的三维坐标
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            cz = results.pose_landmarks.landmark[i].z  # 这里z不需要整数转换

            # 存入数组
            cx_list.append(cx)
            cy_list.append(cy)
            cz_list.append(cz)

        if max(cy_list[25], cy_list[26]) < min(cy_list[11], cy_list[12]):
            rating = 1
            img = cv2.putText(img, 'Supine', (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
    

    else:
        scaler = 1
        failure_str = 'No Person'
        img = cv2.putText(img, failure_str, (25 * scaler, 100 * scaler), cv2.FONT_HERSHEY_SIMPLEX, 1.25 * scaler, (255, 0, 255), 2 * scaler)
        # print('从图像中未检测出人体关键点，报错。')
    
    return img, rating


# 视频分析函数（返回视频的帧数）
def process_video(video_path):

    base_name = os.path.basename(video_path)  # 只获取文件名
    output_filename = f"output_{base_name}"  # 给文件名加 "output_" 前缀
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)  # 组合完整路径

    print('视频开始处理',video_path)
    
    # 获取视频总帧数
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        return "视频无效，帧数为0"
    print('视频总帧数为:', frame_count)
    
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

    total_rating = 0
    
    # 进度条绑定视频总帧数
    with tqdm(total=frame_count-1) as pbar:
        try:
            while(cap.isOpened()):
                success, frame = cap.read()
                if not success:
                    break

                try:
                    frame, frame_rating = process_frame(frame)
                except:
                    print('error')
                    frame_rating = 0
                
                total_rating += frame_rating
                out.write(frame)

                # 进度条更新一帧
                pbar.update(1)

        except Exception as e:
            print(f'中途中断: {e}')
            pass

    cv2.destroyAllWindows()
    out.release()
    cap.release()

    video_score = total_rating / frame_count * 100 if frame_count > 0 else 0

    print(f'视频评分: {video_score:.2f}%')
    print('视频已保存:', output_path)

    return f"视频评分: 其中占比{video_score:.2f}%的帧数视觉受限"


@app.route('/analyze-url', methods=['POST'])
def analyze_video():
    # 获取上传的视频文件
    if 'video' not in request.files:
        return jsonify({'result': '没有上传视频文件！'}), 400

    video = request.files['video']

    # 确保文件名不是空的
    if video.filename == '':
        return jsonify({'result': '未选择视频文件！'}), 400

    # 保存视频文件到上传目录
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    try:
        # 处理视频
        result = process_video(video_path)

        # 返回处理结果
        return jsonify({'result': result})

    finally:
         # 删除上传的视频文件及其所在目录
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f'已删除文件: {video_path}')

    #     # 如果文件夹为空，删除该目录
    #     if not os.listdir(UPLOAD_FOLDER):
    #         shutil.rmtree(UPLOAD_FOLDER)
    #         print(f'已删除空目录: {UPLOAD_FOLDER}')

if __name__ == '__main__':
    app.run(debug=True)