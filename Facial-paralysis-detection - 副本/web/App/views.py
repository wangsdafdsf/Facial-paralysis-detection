from flask import Blueprint, render_template, request, jsonify, redirect, url_for
import base64
import io
from PIL import Image
import torch
import torch.nn.functional as F
from model.ViTforFaceParalysisDetection.predict import predict_image_from_memory, transform
from model.ViTforFaceParalysisDetection.ViTforFaceParalysisDetection import ViTForFaceParalysisDetection
import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import numpy as np
from .utils  import generate_paralysis_report


blue = Blueprint('User', __name__)


@blue.route('/')
def index():
    return render_template('index.html')


@blue.route('/newConversations', methods=['GET', 'POST'])
def new_conversations():
    return render_template('index.html')


@blue.route('/currentConversations', methods=['GET', 'POST'])
def current_conversations():
    from . import db
    from .models import Conversation
    if request.method == 'POST':
        user_input = request.form.get('message')
        if user_input:
            model_response = f"这是对 '{user_input}' 的回复（大模型处理结果）"

            conversation = Conversation(
                user_input=user_input,
                model_response=model_response
            )
            db.session.add(conversation)
            db.session.commit()

    conversations = Conversation.query.order_by(Conversation.created_at).all()
    return render_template('currentConversations.html', conversations=conversations)


@blue.route('/delete_all_conversations', methods=['POST'])
def delete_all_conversations():
    from . import db
    from .models import Conversation

    Conversation.query.delete()
    db.session.commit()

    return redirect(url_for('User.current_conversations'))



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViTForFaceParalysisDetection(
    image_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=2,
    embed_dim=768,
    depth=12,
    num_heads=16,
    mlp_dim=3072,
    dropout=0.5,
).to(device)
model_path = os.path.join(os.path.dirname(__file__), '../../model/ViTforFaceParalysisDetection/best_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

main = Blueprint('main', __name__)


@main.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "未上传图片"}), 400

        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        vit_result = predict_image_from_memory(
            image=image,
            model=model,
            transform=transform,
            device=device
        )

        deepseek_report = generate_paralysis_report(vit_result)
        return jsonify({
            "vit_result": vit_result,
            "deepseek_report": deepseek_report
        })
    except Exception as e:
        current_app.logger.error(f"预测过程出错: {str(e)}")
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500

