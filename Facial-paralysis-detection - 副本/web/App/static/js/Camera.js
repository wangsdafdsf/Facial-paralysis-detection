document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('videoElement');
    const permissionSection = document.getElementById('permission-section');
    const videoContainer = document.getElementById('video-container');
    const permissionOverlay = document.getElementById('permission-overlay');
    const errorSection = document.getElementById('error-section');
    const errorMessage = document.getElementById('error-message');
    const requestButton = document.getElementById('request-camera');


    videoContainer.style.display = 'none';

    let retryCount = 0;
    const maxRetries = 3;

    requestButton.addEventListener('click', requestCamera);

    function requestCamera() {
        permissionSection.style.display = 'none';
        videoContainer.style.display = 'block';
        permissionOverlay.style.display = 'flex';

        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            .then(function(stream) {
                permissionOverlay.style.display = 'none';
                video.srcObject = stream;
                video.play();
                retryCount = 0;
            })
            .catch(function(err) {
                console.error("摄像头访问出错: " + err);
                videoContainer.style.display = 'none';
                errorSection.style.display = 'block';

                if (err.name === 'NotAllowedError') {
                    errorMessage.textContent = "请在浏览器设置中允许摄像头访问权限";
                } else if (err.name === 'NotFoundError') {
                    errorMessage.textContent = "未检测到可用摄像头";
                } else {
                    errorMessage.textContent = `摄像头访问失败: ${err.message}`;
                }

                let retryButton = document.querySelector('#error-section button');
                if (!retryButton) {
                    retryButton = document.createElement('button');
                    retryButton.textContent = '重试';
                    retryButton.addEventListener('click', function() {
                        errorSection.style.display = 'none';
                        if (retryCount < maxRetries) {
                            retryCount++;
                            requestCamera();
                        } else {
                            permissionSection.style.display = 'block';
                            retryCount = 0;
                        }
                    });
                    errorSection.appendChild(retryButton);
                }
            });
    }

 document.getElementById('capture-btn').addEventListener('click', async () => {
    const video = document.getElementById('videoElement');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('image', blob, 'capture.jpg');

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `服务器错误: ${response.status}`);
            }

            const result = await response.json();

            const vitResult = result.vit_result;
            document.getElementById('diagnosis').textContent = `诊断结果：${vitResult.diagnosis_result}`;
            document.getElementById('confidence').textContent = `可信度：${(vitResult.confidence * 100).toFixed(1)}%`;

            const reportContainer = document.getElementById('deepseekReport');
            const reportContent = document.getElementById('reportContent');
            reportContainer.style.display = 'block';
            reportContent.textContent = result.deepseek_report;

        } catch (error) {
            console.error("检测失败详情：", error);
            const resultDiv = document.getElementById('result');
            resultDiv.style.color = 'red';
            resultDiv.textContent = `检测失败：${error.message}，请重试`;

            document.getElementById('deepseekReport').style.display = 'none';
        }
    });
});
});