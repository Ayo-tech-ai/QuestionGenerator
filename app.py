import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Cache model loading to avoid repeated downloads
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Load the model and tokenizer
tokenizer, model = load_model()

# Streamlit app UI
st.set_page_config(page_title="AI Question Generator", layout="wide")
st.title("🧠 AI Question Generator for Teachers")
st.write("Paste a passage below, and this app will generate fact-based comprehension questions using a large language model.")

# Input: passage
passage = st.text_area("📄 Paste your comprehension passage here:", height=300)

# Input: number of questions
num_questions = st.slider("🧮 Number of questions to generate", min_value=1, max_value=10, value=5)

# Action: generate button
if st.button("🚀 Generate Questions"):
    if not passage.strip():
        st.warning("⚠️ Please paste a passage before clicking Generate.")
    else:
        st.success("Generating questions... Please wait.")
        generated_questions = []

        for i in range(num_questions):
            prompt = (
                f"Based strictly on the passage below, generate 1 clear and fact-based comprehension question. "
                f"Avoid repetition. Do not invent any facts. Each question should test understanding of a different part of the passage.\n\n{passage}"
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
            generated_questions.append(question.strip())

        # Output: Display generated questions
        st.markdown("### ✅ Generated Questions:")
        for idx, q in enumerate(generated_questions, 1):
            st.markdown(f"**{idx}. {q}**")
