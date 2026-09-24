from datetime import datetime
from . import db


class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_input = db.Column(db.Text, nullable=False)
    model_response = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_input': self.user_input,
            'model_response': self.model_response,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }