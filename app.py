import os
import re  # for cleaning text
import streamlit as st
from transformers import pipeline

# App title
st.title("🍰 Recipe Generator using Fine-Tuned GPT-2")

# Load model and tokenizer
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # folder containing app.py
    model_path = os.path.join(base_dir, "gpt2-recipes-final")  # folder with model files

    generator = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=model_path,
        use_fast=False  # force slow tokenizer to avoid ModelWrapper errors
    )
    return generator

generator = load_model()

# Input fields
title = st.text_input("Enter recipe title:", "Simple Pancakes")
ingredients = st.text_area("Enter ingredients (comma separated):", "eggs, milk, flour, sugar")

if st.button("Generate Recipe"):
    # Structured prompt
    prompt = (
        f"TITLE: {title}\n"
        f"INGREDIENTS: {ingredients}\n\n"
        "STEPS:\n1."
    )

    # Generate recipe
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

    # Extract text
    recipe = outputs[0]["generated_text"]

    # ---------------- Post-processing ----------------
    # 1. Remove URLs
    recipe_clean = re.sub(r"http\S+", "", recipe)
    # 2. Remove stray symbols
    recipe_clean = re.sub(r'[\]\["\>\<]+', '', recipe_clean)
    # 3. Strip extra whitespace
    recipe_clean = recipe_clean.strip()

    # Display
    st.subheader("Generated Recipe:")
    st.text(recipe_clean)
