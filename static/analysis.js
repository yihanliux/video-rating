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
                document.getElementById("result-text").textContent = data.result;
                document.getElementById("result-text").style.display = "block";
            } else {
                setTimeout(checkStatus, 2000);
            }
        });
    }
    checkStatus();
});
