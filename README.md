# 🍳 Recipe Generator using Fine-Tuned GPT-2

## Problem Statement

Generate cooking recipes given either a list of ingredients or a recipe title.  
The goal is to create coherent, creative, and human-readable recipes automatically using a generative language model.

---

## Dataset

We used the **[Kaggle 3A2MExt Recipe Dataset](https://www.kaggle.com/datasets/nazmussakibrupol/3a2mext/data)**.  

- For this project, the **first 100k rows** of the dataset were utilized.  
- Each record contains:
  - Recipe title  
  - Ingredients list  
  - Recipe steps  
  - Category (optional)

---

## Objective

- Fine-tune **GPT-2** to generate recipes.  
- **Input:** Recipe title or ingredients list  
- **Output:** Full recipe with step-by-step instructions

---

## Project Deliverables

### 1. Tokenization & Dataset Formatting Script
- Preprocesses raw CSV data into the format required for GPT-2.  
- Handles tokenization of recipe titles, ingredients, and steps.

### 2. Training Loop for GPT-2
- Fine-tunes GPT-2 on the dataset.  
- Supports text generation with `max_length`, `temperature`, `top_k`, and `top_p` sampling parameters.

### 3. Example Generations & Quality Evaluation
- Generated sample recipes using the fine-tuned model.  
- Evaluated outputs using:
  - **ROUGE** (Rouge-1, Rouge-2, Rouge-L)  
  - **BLEU** scores  
  - Optional human evaluation for coherence and creativity.

### 4. Interactive Application
- **Streamlit App** (or Gradio) for end-users to generate recipes interactively.  
- Input options:
  - Recipe title  
  - Comma-separated list of ingredients  
- Outputs step-by-step recipes with clean formatting.

---

## Example Usage

**Input:**  
Title: Simple Pancakes
Ingredients: eggs, milk, flour, sugar


**Output:**  


In a bowl, whisk eggs and milk together.

Gradually add flour and sugar, mixing until smooth.

Heat a pan over medium heat, pour batter, and cook pancakes until golden.

Serve warm with syrup or toppings of choice.


---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/ibraheem8887/Recipe_Generator.git
cd Recipe_Generator


Install dependencies:

pip install -r requirements.txt


Launch the Streamlit app:

streamlit run app.py


