// 测试API连接和Axios是否正常工作
console.log('测试脚本加载成功');

// 检查Axios是否已加载
if (typeof axios !== 'undefined') {
    console.log('Axios已成功加载');
    console.log('API连接测试已禁用，以避免页面加载时的错误弹窗');
} else {
    console.error('Axios未加载成功');
}