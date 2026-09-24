// 输入框聚焦效果
const inputField = document.querySelector('.input-panel form input[type="text"]');
inputField.addEventListener('focus', function () {
    this.placeholder = '';
});

inputField.addEventListener('blur', function () {
    this.placeholder = '请输入您的问题...';
});