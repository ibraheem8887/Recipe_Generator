import os
import streamlit as st
from transformers import pipeline
import re  # for cleaning text

# ----------------------------
# App title
# ----------------------------
st.title("🍰 Recipe Generator using Fine-Tuned GPT-2")

# ----------------------------
# Load model and tokenizer
# ----------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)  # folder containing app.py
    model_path = os.path.join(base_dir, "../gpt2-recipes-final")  # sibling folder

    generator = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=model_path,
        use_fast=False  # force slow tokenizer to avoid ModelWrapper error
    )
    return generator

generator = load_model()

# ----------------------------
# Input fields
# ----------------------------
title = st.text_input("Enter recipe title:", "Simple Pancakes")
ingredients = st.text_area("Enter ingredients (comma separated):", "eggs, milk, flour, sugar")

# ----------------------------
# Generate recipe
# ----------------------------
if st.button("Generate Recipe"):
    # Structured prompt
    prompt = (
        f"TITLE: {title}\n"
        f"INGREDIENTS: {ingredients}\n\n"
        "STEPS:\n1."
    )

    # Generation with improved parameters
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

    recipe = outputs[0]["generated_text"]

    # ----------------------------
    # Post-processing to clean output
    # ----------------------------
    recipe_clean = re.sub(r"http\S+", "", recipe)       # remove URLs
    recipe_clean = re.sub(r'[\]\["\>\<]+', '', recipe_clean)  # remove stray symbols
    recipe_clean = recipe_clean.strip()                # strip extra whitespace

    st.subheader("Generated Recipe:")
    st.markdown(recipe_clean)  # allows line breaks and better formatting
