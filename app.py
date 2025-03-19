from flask import Flask, render_template, request, jsonify
import time
import os

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
    time.sleep(5)  # 模拟处理时间
    return jsonify({"done": True, "result": "Analysis Complete!"})

if __name__ == "__main__":
    app.run(debug=True)