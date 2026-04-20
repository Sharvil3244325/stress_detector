import gradio as gr
from transformers import pipeline

# ================= LOAD MODEL =================
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")

# ================= FUNCTIONS =================

def analyze_text(text):
    result = classifier(text)[0]
    return result['label'], result['score']


def lifestyle_score(sleep, study_hours):
    score = 0
    
    if sleep < 6:
        score += 1
    if study_hours > 8:
        score += 1

    return score


def final_prediction(text, sleep, study):
    label, confidence = analyze_text(text)
    life_score = lifestyle_score(sleep, study)

    if label == "NEGATIVE" or life_score >= 1:
        status = "⚠️ High Stress"
        color = "red"
        tip = "Take breaks, sleep well, and relax your mind 🧘‍♂️"
    else:
        status = "✅ Low Stress"
        color = "green"
        tip = "Great job! Keep maintaining your routine 💪"

    return f"""
### {status}

**🧠 Sentiment:** {label}  
**📊 Confidence:** {round(confidence*100,2)}%  
**😴 Sleep:** {sleep} hrs  
**📚 Study:** {study} hrs  

---

💡 **Advice:** {tip}
"""


# ================= CUSTOM CSS =================

custom_css = """
body {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
}
.gradio-container {
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    text-align: center;
    color: white;
}
textarea {
    border-radius: 10px !important;
}
"""


# ================= UI =================

with gr.Blocks(css=custom_css) as demo:
    
    gr.Markdown("# 🧠 AI Stress Detector")
    gr.Markdown("### 💙 Check your mental health using AI + lifestyle analysis")

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                label="📝 How are you feeling today?",
                placeholder="Type your thoughts here...",
                lines=4
            )

            sleep = gr.Slider(0, 12, value=6, label="😴 Sleep Hours")
            study = gr.Slider(0, 12, value=4, label="📚 Study Hours")

            btn = gr.Button("🔍 Analyze Stress", variant="primary")

        with gr.Column():
            output = gr.Markdown()

    btn.click(fn=final_prediction, inputs=[text, sleep, study], outputs=output)

# ================= RUN =================

if __name__ == "__main__":
    demo.launch()