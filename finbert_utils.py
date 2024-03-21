# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple

# Check if CUDA is available, use GPU if available, otherwise use CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load pre-trained tokenizer and model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

# Define sentiment labels
labels = ["positive", "negative", "neutral"]

# Function to estimate sentiment of news articles
def estimate_sentiment(news):
    if news:
        # Tokenize the news articles
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)
    
        # Pass tokenized input through the model
        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
        
        # Softmax to convert logits to probabilities
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1) 
        
        # Get the maximum probability and corresponding sentiment label
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        return probability, sentiment
    else:
        # If no news articles provided, return neutral sentiment
        return 0, labels[-1]

# Main function
if __name__ == "__main__":
    # Example usage of the estimate_sentiment function
    tensor, sentiment = estimate_sentiment(['markets responded positively to the news!', 'traders were pleasantly surprised!!'])
    
    # Print the sentiment probability and label
    print(tensor, sentiment)
    
    # Print whether CUDA (GPU) is available
    print("CUDA Available:", torch.cuda.is_available())
