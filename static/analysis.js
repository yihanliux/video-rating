document.addEventListener("DOMContentLoaded", function () {
    console.log("✅ Running analysis page script...");

    // 获取视频 URL
    fetch("/get_video")
    .then(response => response.json())
    .then(data => {
        console.log("✅ Server response from /get_video:", data);  // 🔍 检查 Flask 返回的 JSON

        if (data.video_url) {
            console.log("✅ Setting video source to:", data.video_url);
            const videoPlayer = document.getElementById("videoPlayer");
            videoPlayer.src = data.video_url;
            videoPlayer.load();
        } else {
            console.error("❌ No video URL received from server!");
        }
    })
    .catch(error => {
        console.error("❌ Error fetching video URL:", error);
    });

    function checkStatus() {
        fetch("/check_status")
        .then(response => response.json())
        .then(data => {
            if (data.done) {
                document.getElementById("loading-text").style.display = "none";
                document.querySelector(".loader").style.display = "none";
                document.getElementById("result-text").innerHTML = data.result.replace(/\n/g, "<br>");
                document.getElementById('result-container').style.display = 'flex';

                // 更新图片
                const resultImage = document.getElementById("result-image");
                resultImage.src = data.image_url;
                resultImage.style.display = 'block';

                const resultImage1 = document.getElementById("suggestion-image-1");
                resultImage1.src = data.image_url_1;
                resultImage1.style.display = 'block';

                const resultImage2 = document.getElementById("suggestion-image-2");
                if (data.image_url_2 !== "None") {  // 确认 image_url_2 不是 None 或 null
                    resultImage2.src = data.image_url_2;
                    resultImage2.style.display = 'block';
                } else {
                    resultImage2.style.display = 'none';  // 如果是 None 或 null，则不展示图片
                }
            } else {
                setTimeout(checkStatus, 2000);
            }
        });
    }
    checkStatus();
});
