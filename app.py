from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, render_template_string

app = Flask(__name__)

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot</title>
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            display: flex;
            height: 100vh;
            background-color: #f0f0f0;
        }
        .container {
            display: flex;
            width: 100%;
            height: 100%;
        }
        .sidebar {
            width: 300px;
            background-color: #f4f4f4;
            padding: 20px;
            box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
        }
        .sidebar h2 {
            margin-top: 0;
        }
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            background-color: #fff;
        }
        #welcome-message {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 20px;
        }
        #chat-box {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        #messages {
            flex: 1;
            padding: 10px;
            overflow-y: auto;
            border-top: 1px solid #ddd;
        }
        #input-area {
            display: flex;
            padding: 10px;
            border-top: 1px solid #ddd;
        }
        #prompt {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            padding: 10px 20px;
            border: none;
            background-color: #007bff;
            color: white;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>Chat History</h2>
            <ul id="history">
                <!-- Placeholder for chat history -->
            </ul>
        </div>
        <div class="main-content">
            <div id="welcome-message">
                <h1>Welcome to the Chatbot</h1>
                <p>Enter your message below to start chatting.</p>
            </div>
            <div id="chat-box">
                <div id="messages">
                    {% if response %}
                        <div><strong>Bot:</strong> {{ response }}</div>
                    {% endif %}
                    {% if error_message %}
                        <div style="color: red;"><strong>Error:</strong> {{ error_message }}</div>
                    {% endif %}
                </div>
                <form method="post">
                    <div id="input-area">
                        <input type="text" id="prompt" name="prompt" placeholder="Type your message here..." required>
                        <button type="submit">Send</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    response = ""
    error_message = ""
    if request.method == 'POST':
        prompt = request.form.get('prompt', '')
        if prompt:
            try:
                input_ids = tokenizer.encode(prompt, return_tensors='pt')
                output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

                response = tokenizer.decode(output[0], skip_special_tokens=True)
            except Exception as e:
                error_message = str(e)
    return render_template_string(html_template, response=response, error_message=error_message)

if __name__ == '__main__':
    app.run()
