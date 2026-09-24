import requests
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../api.env'))
DASHCOPE_API_KEY = os.getenv("DASHCOPE_API_KEY")


def generate_paralysis_report(vit_result):
    if not DASHCOPE_API_KEY:
        return "错误：未配置通义千问API密钥（DASHCOPE_API_KEY）"

    prompt = f"""以下是人脸偏瘫检测模型的分析结果：
- 诊断结论：{vit_result['diagnosis_result']}
- 可信度：{vit_result['confidence']}（即{vit_result['confidence'] * 100:.1f}%）
- 关键特征：{vit_result.get('key_features', '无')}

请生成一份面向普通用户的检测报告，要求：
1. 结果总结：用通俗语言解释结果和可信度含义；
2. 建议措施：偏瘫则建议就医科室+日常注意事项；无偏瘫则建议健康习惯；
3. 免责提示：明确说明仅为辅助检测，不替代专业医生诊断。
语言温和、专业，避免引起恐慌。
"""

    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {DASHCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen-turbo",
        "input": {
            "messages": [{"role": "user", "content": prompt}]
        },
        "parameters": {
            "temperature": 0.2,
            "max_tokens": 500
        }
    }

    try:
        response = requests.post(
            url=url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        if "output" in result and "text" in result["output"]:
            report = result["output"]["text"]
            return report
        else:
            return f"通义千问返回格式异常：{str(result)}"

    except requests.exceptions.HTTPError as e:
        return f"通义千问调用失败：{response.text}"
    except requests.exceptions.Timeout:
        return "通义千问调用失败：请求超时"
    except Exception as e:
        return f"通义千问调用异常：{str(e)}"