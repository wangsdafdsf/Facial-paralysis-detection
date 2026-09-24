document.addEventListener('DOMContentLoaded', function() {
    // 处理表单提交，使用AJAX避免页面刷新
    const form = document.querySelector('.input-panel form');
    const conversationHistory = document.getElementById('conversation-history');

    if (form && conversationHistory) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();

            const input = form.querySelector('input[name="message"]');
            const userInput = input.value.trim();

            if (userInput) {
                // 先在界面上显示用户输入
                const userMessageDiv = document.createElement('div');
                userMessageDiv.className = 'message user';
                userMessageDiv.innerHTML = `
                    <div class="message-content">
                        <p>${userInput}</p>
                        <small class="time">${new Date().toLocaleString()}</small>
                    </div>
                `;
                conversationHistory.appendChild(userMessageDiv);
                conversationHistory.scrollTop = conversationHistory.scrollHeight;

                // 显示加载状态
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'message ai loading';
                loadingDiv.innerHTML = `
                    <div class="message-content">
                        <p>思考中...</p>
                    </div>
                `;
                conversationHistory.appendChild(loadingDiv);
                conversationHistory.scrollTop = conversationHistory.scrollHeight;

                // 发送数据到后端
                fetch(form.action, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `message=${encodeURIComponent(userInput)}`
                })
                .then(response => response.json())
                .then(data => {
                    // 移除加载状态
                    conversationHistory.removeChild(loadingDiv);

                    // 添加AI回复
                    const aiMessageDiv = document.createElement('div');
                    aiMessageDiv.className = 'message ai';
                    aiMessageDiv.innerHTML = `
                        <div class="message-content">
                            <p>${data.response}</p>
                        </div>
                    `;
                    conversationHistory.appendChild(aiMessageDiv);
                    conversationHistory.scrollTop = conversationHistory.scrollHeight;
                })
                .catch(error => {
                    console.error('提交失败:', error);
                    // 移除加载状态并显示错误
                    conversationHistory.removeChild(loadingDiv);
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'message ai';
                    errorDiv.innerHTML = `
                        <div class="message-content" style="color: red;">
                            <p>抱歉，处理请求时出错，请重试</p>
                        </div>
                    `;
                    conversationHistory.appendChild(errorDiv);
                });
            }

            // 清空输入框
            input.value = '';
        });
    }
});