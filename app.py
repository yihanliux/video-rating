
from flask import Flask, request, jsonify, render_template

import os, cv2, shutil

app = Flask(__name__)


# 确保上传文件的文件夹存在
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/')
def home():
    return render_template('index.html')


# 视频分析函数（返回视频的帧数）
def process_video(file_path):
    # 使用 OpenCV 加载视频并计算总帧数
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return "无法打开视频文件"
    
    # 获取视频总帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 释放视频资源
    cap.release()

    return f"视频总帧数: {frame_count}"


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

        # 如果文件夹为空，删除该目录
        if not os.listdir(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            print(f'已删除空目录: {UPLOAD_FOLDER}')

if __name__ == '__main__':
    app.run(debug=True)