import os
import streamlit as st
from transformers import pipeline

st.title("🍰 Recipe Generator using Fine-Tuned GPT-2")

@st.cache_resource
def load_model():
    # Absolute path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "gpt2-recipes-final")

    # Use slow tokenizer to avoid ModelWrapper issues
    generator = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=model_path,
        use_fast=False
    )
    return generator

generator = load_model()

title = st.text_input("Enter recipe title:", "Simple Pancakes")
ingredients = st.text_area("Enter ingredients (comma separated):", "eggs, milk, flour, sugar")

if st.button("Generate Recipe"):
    prompt = f"TITLE: {title}\nINGREDIENTS: {ingredients}\nSTEPS:\n1."
    outputs = generator(
        prompt,
        max_length=250,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        repetition_penalty=1.2
    )
    st.text(outputs[0]["generated_text"])
