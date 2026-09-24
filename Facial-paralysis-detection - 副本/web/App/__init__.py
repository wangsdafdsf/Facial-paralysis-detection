from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def create_app():
    app = Flask(__name__)
    # 配置数据库路径
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///conversations.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # 初始化db
    db.init_app(app)

    from .models import Conversation

    # 延迟导入views并注册蓝图（同时注册main蓝图）
    from .views import blue, main  # 导入main蓝图
    app.register_blueprint(blue)
    app.register_blueprint(main)  # 注册main蓝图


    with app.app_context():
        db.create_all()

    return app