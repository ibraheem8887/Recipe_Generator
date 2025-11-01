from transformers import AutoTokenizer

# Path where your trained model is stored
model_path = r"D:\GenAi\Recipe-generator\gpt2-recipes-final"

print("⏳ Downloading clean GPT-2 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print("💾 Saving tokenizer to your model folder...")
tokenizer.save_pretrained(model_path)

print("✅ Tokenizer fixed successfully!")
