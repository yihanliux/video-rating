document.getElementById('linkForm').addEventListener('submit', function(event) {
    event.preventDefault(); // 防止表单默认提交行为

    const videoInput = document.getElementById('videoInput').files[0]; // 获取用户选择的文件
    const resultElement = document.getElementById('result');

    // 确保用户选择了文件
    if (videoInput) {
        const formData = new FormData();
        formData.append('video', videoInput); // 将视频文件添加到表单数据中

        // 向后端发送文件
        fetch('http://127.0.0.1:5000/analyze-url', {
            method: 'POST',
            body: formData // 将 FormData 作为请求体发送
        })
        .then(response => response.json()) // 解析返回的 JSON
        .then(data => {
            resultElement.textContent = data.result; // 显示后端返回的结果
        })
        .catch(error => {
            resultElement.textContent = "请求出错！";
            console.error("Error:", error);
        });
    } else {
        resultElement.textContent = "请选择一个视频文件！";
    }
});
