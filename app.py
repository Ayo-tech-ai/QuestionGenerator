import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Cache the model to avoid reloading every time
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# App layout
st.set_page_config(page_title="AI Question Generator", layout="wide")
st.title("🧠 AI Question Generator for Teachers")
st.markdown("""
Welcome to your personal **AI-powered comprehension question assistant**. Paste a short passage, and this app will generate questions intelligently based on different parts of the text.

---

### ⚠️ Important Notes for Teachers:
- This is an **AI tool** — it's powerful, but not perfect. Always review what it generates.
- You can use the questions as **theory** or **multiple choice**.
- You should still **decide how to use the questions** and **add correct answers** where needed.
- Keep passages **short and focused**. Max limit: **400 words**.
---
""")

# Text input
passage = st.text_area("📄 Paste your comprehension passage here:", height=300)

# Word limit enforcement
word_count = len(passage.strip().split())
if word_count > 400:
    st.error(f"❌ Passage is too long: {word_count} words (maximum is 400). Please shorten it.")
    st.stop()

# Question count slider (max 5)
num_questions = st.slider("🔢 Number of questions to generate:", min_value=1, max_value=5, value=3)

if st.button("🚀 Generate Questions"):
    if not passage.strip():
        st.warning("⚠️ Please enter a passage first.")
        st.stop()

    # Split into words
    words = passage.strip().split()
    chunk_size = len(words) // num_questions

    chunks = [
        " ".join(words[i * chunk_size : (i + 1) * chunk_size])
        for i in range(num_questions)
    ]

    # Adjust final chunk to include leftovers
    if len(words) % num_questions != 0:
        chunks[-1] += " " + " ".join(words[num_questions * chunk_size :])

    st.success("✅ Questions generated below:")

    for idx, chunk in enumerate(chunks):
        prompt = (
            f"Based on the following passage chunk, generate 1 clear, fact-based comprehension question. "
            f"Avoid repetition and do not invent facts:\n\n{chunk}"
        )

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            num_return_sequences=1
        )

        question = tokenizer.decode(output[0], skip_special_tokens=True)
        st.markdown(f"**{idx+1}. {question.strip()}**")
