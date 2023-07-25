from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import torch

app = Flask(__name__)

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # Set the model to evaluation mode
@app.route('/api/similarity', methods=['POST'])
def calculate_similarity():
    # Get the text pairs from the request's JSON data
    data = request.get_json()
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')

    # Process the text and calculate embeddings
    with torch.no_grad():
        inputs = tokenizer([text1, text2], return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        embeddings1 = outputs.last_hidden_state[0].mean(dim=0)  # Average pooling over tokens
        embeddings2 = outputs.last_hidden_state[1].mean(dim=0)  # Average pooling over tokens

    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=0).item()

    # Return the similarity score as a JSON response
    return jsonify({'similarity_score': similarity_score})
if __name__ == '__main__':
    app.run(debug=True)



