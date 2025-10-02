from src.model import FAQModel
from src.preprocessing import preprocess

# src/chatbot.py
from src.preprocessing import preprocess

class Chatbot:
    def __init__(self, model):
        self.model = model

    def chatbot_response(self, message: str) -> str:
        """Return the best matching response for a given message."""
        processed = preprocess(message)
        return self.model.get_response(processed)