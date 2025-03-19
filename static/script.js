document.addEventListener("DOMContentLoaded", function () {
    console.log("✅ script.js loaded!");

    const uploadButton = document.getElementById("submitBtn");
    const dropZone = document.getElementById("dropZone");

    if (!uploadButton) {
        console.error("❌ submitBtn 未找到！（这可能是 analysis.html 页面）");
        return;
    }

    console.log("✅ submitBtn 已找到！");

    const videoInput = document.createElement("input");
    videoInput.type = "file";
    videoInput.accept = "video/*";
    videoInput.style.display = "none";

    uploadButton.addEventListener("click", function () {
        console.log("✅ Upload button clicked!");
        videoInput.click();
    });

    videoInput.addEventListener("change", function () {
        if (videoInput.files.length > 0) {
            console.log("✅ File selected:", videoInput.files[0].name);
            uploadVideo(videoInput.files[0]);
        }
    });

        // ✅ 监听拖拽事件
    dropZone.addEventListener("dragover", function (event) {
        event.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", function () {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", function (event) {
        event.preventDefault();
        dropZone.classList.remove("drag-over");

        const file = event.dataTransfer.files[0];
        if (file) {
            console.log("✅ File dropped:", file.name);
            uploadVideo(file);
        }
    });
    

    function uploadVideo(file) {
        console.log("✅ Uploading file:", file.name);

        const formData = new FormData();
        formData.append("video", file);

        fetch("/upload", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log("✅ Server response:", data);
            if (data.status === "success") {
                window.location.href = "/analysis";
            } else {
                alert("❌ Upload failed. Please try again.");
            }
        })
        .catch(error => {
            console.error("❌ Upload error:", error);
            alert("❌ An error occurred while uploading.");
        });
    }
});

document.querySelectorAll(".disabled-link").forEach(link => {
    link.addEventListener("click", function (event) {
        event.preventDefault(); // 禁止跳转
    });
});




