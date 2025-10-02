import os
import tkinter as tk
from src.model import FAQModel

# Force CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_PATH = "models/model.pkl"

# Load or train model
if not os.path.exists(MODEL_PATH):
    print("⚡ Training new model...")
    model = FAQModel().train(epochs=300)
    model.save(MODEL_PATH)
else:
    print("Loading existing model...")
    model = FAQModel.load(MODEL_PATH)


# ---------------- GUI ---------------- #
class ChatbotGUI:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("Chatbot")
        self.root.geometry("500x600")

        # Chat history
        self.chat_log = tk.Text(root, bg="white", fg="black", font=("Arial", 12))
        self.chat_log.config(state=tk.DISABLED, wrap=tk.WORD)
        self.chat_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Add tags for alignment
        self.chat_log.tag_configure("user", foreground="blue", justify="left")
        self.chat_log.tag_configure("bot", foreground="green", justify="right")

        # Entry box
        self.entry = tk.Entry(root, font=("Arial", 12))
        self.entry.pack(padx=10, pady=5, fill=tk.X)
        self.entry.bind("<Return>", self.send)

        # Send button
        self.send_button = tk.Button(root, text="Send", command=self.send)
        self.send_button.pack(pady=5)

    def send(self, event=None):
        msg = self.entry.get().strip()
        if not msg:
            return
        self.entry.delete(0, tk.END)

        # Display user message (left)
        self.chat_log.config(state=tk.NORMAL)
        self.chat_log.insert(tk.END, "You: " + msg + "\n", "user")
        self.chat_log.config(state=tk.DISABLED)
        self.chat_log.yview(tk.END)

        # Get bot response
        res = self.model.get_response(msg)
        if isinstance(res, list):
            import random
            res = random.choice(res)

        # Display bot message (right)
        self.chat_log.config(state=tk.NORMAL)
        self.chat_log.insert(tk.END, "Bot: " + res + "\n\n", "bot")
        self.chat_log.config(state=tk.DISABLED)
        self.chat_log.yview(tk.END)


# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    gui = ChatbotGUI(root, model)
    root.mainloop()
