import os
import torch
from torchvision import transforms
from PIL import Image
from model.ViTforFaceParalysisDetection.ViTforFaceParalysisDetection import ViTForFaceParalysisDetection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViTForFaceParalysisDetection().to(device)

model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)

model.eval()

# 预处理方法
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



def predict_image(image_path, model, transform, device):
    """
    预测单张图像的偏瘫结果，并返回结构化信息（含置信度）

    参数:
        image_path: str，图像文件路径
        model: 训练好的ViT模型
        transform: 图像预处理管道（与训练/验证一致）
        device: 运行设备（如'cuda'或'cpu'）
    返回:
        dict: 结构化结果，包含：
            - diagnosis_result: str，'normal'或'paralysis'
            - confidence: float，置信度（0-1之间）
            - key_features: str，暂为'未提供'（若模型不输出特征）
    """
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # 增加batch维度并移至设备

    # 模型推理（关闭梯度计算，提高速度）
    with torch.no_grad():
        outputs = model(image)  # 模型输出logits（未经过softmax的原始分数）

        # 计算置信度（通过softmax将logits转换为概率）
        probs = torch.nn.functional.softmax(outputs, dim=1)  # 对输出做softmax，得到概率分布
        confidence, predicted = torch.max(probs, 1)  # 取最大概率及对应的类别索引

    # 映射类别索引到标签
    classes = ['normal', 'paralysis']
    predicted_class = classes[predicted.item()]  # 预测的类别（'normal'或'paralysis'）
    confidence_value = confidence.item()  # 置信度（0-1之间，如0.92表示92%）

    # 返回结构化字典
    return {
        "diagnosis_result": predicted_class,
        "confidence": confidence_value,
        "key_features": "未提供"  # 若模型不输出特征，可暂用此值；若有特征可替换
    }


# 使用与验证集相同的预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 测试预测
# image_path = "data/test/a-32-220x300.jpg"
# prediction = predict_image(image_path, model, transform, device)
# print(f"预测结果: {prediction}")

def predict_image_from_memory(image, model, transform, device):
    """
    从内存中的PIL图像预测（适配Web上传的图像，无需保存文件）

    参数:
        image: PIL.Image对象，内存中的图像（从Web上传的字节流转换而来）
        model: 训练好的ViT模型
        transform: 预处理管道
        device: 运行设备
    返回:
        dict: 同predict_image的结构化结果
    """
    # 直接对内存图像做预处理（无需从路径加载）
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    classes = ['normal', 'paralysis']
    predicted_class = classes[predicted.item()]
    confidence_value = confidence.item()

    return {
        "diagnosis_result": predicted_class,
        "confidence": confidence_value,
        "key_features": "未提供"
    }