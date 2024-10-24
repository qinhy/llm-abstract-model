
import requests
from LLMAbstractModel.utils import RegxExtractor
from LLMAbstractModel import LLMsStore,Model4LLMs
descriptions = Model4LLMs.Function.param_descriptions
def myprint(string):
    print('##',string,':\n',eval(string),'\n')

store = LLMsStore()
vendor = store.add_new_openai_vendor(api_key="OPENAI_API_KEY")
debug = True

import requests
import os
import json

@descriptions(
    'This function retrieves the most relevant memories from a vector memory store based on a given question. '
    'It uses vector embeddings to find and rank the top N similar memories in the store.',
    question='A string input representing the query or question for which relevant memories need to be retrieved.'
)
class SimpleLLMMemory(Model4LLMs.Function):
    top_n: int = 10

    def __call__(self, question: str):
        # Convert question to vector using OpenAI API
        question_vec = self.vectorize_question(question)
        
        # Search memory vector store for top_n most similar vectors
        memories = self.search_memory(question_vec, top_n=self.top_n)
        
        # Sort memories by similarity score (descending order)
        sorted_memories = sorted(memories, key=lambda x: x['similarity'], reverse=True)
        
        # Return only the most relevant memory data
        top_memories = [memory['data'] for memory in sorted_memories]
        
        return top_memories

    def vectorize_question(self, question: str):
        # Set up the API URL
        url = "https://api.openai.com/v1/embeddings"

        # Set up headers including the OpenAI API key
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }

        # Prepare the data payload for the POST request
        data = {
            "model": "text-embedding-ada-002",  # The embedding model from OpenAI
            "input": question
        }

        # Make the API request to OpenAI
        response = requests.post(url, headers=headers, data=json.dumps(data))

        # Check if the request was successful
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code}, {response.text}")

        # Extract the embedding vector from the response
        embedding = response.json()['data'][0]['embedding']

        return embedding

    def search_memory(self, question_vec, top_n: int):
        # Placeholder for searching vector memory store
        # This would typically involve something like cosine similarity or a nearest neighbor search
        return memory_store.search_by_vector(question_vec, top_n=top_n)

# Make sure to set up your OpenAI API key before using the OpenAI API
# Example: export OPENAI_API_KEY="your-api-key" in the environment

# Add functions for reverse geocoding and address extraction
french_address_search_function = store.add_new_obj(FrenchReverseGeocodeFunction())
