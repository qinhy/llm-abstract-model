import os
from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Model4LLMs

# Initialize the LLMs store
store = LLMsStore()

# Load previously saved LLM configurations from JSON file
store.load('./tmp/ex.1.1HowToUse.json')

# Get the first available LLM model
llm = store.find_all_llms()[0]

# Example 1: Simple question to the LLM
# Send a basic greeting and ask for the LLM's name
print(llm('hi! What is your name?'))
# -> Hello! I'm called Assistant. How can I help you today?

# Example 2: Using system and user messages
# Configure the LLM as an English translator and request translation
print(llm([
    {'role':'system','content':'You are a highly skilled professional English translator.'},
    {'role':'user','content':'"こんにちは！"'}
]))
# -> Hello! I'm an AI language model created by OpenAI, and I don't have a personal name, but you can call me Assistant. How can I help you today?
# -> "Hello!"

# Example 3: Text embedding demonstration
# Get the first available text embedding model and generate embeddings
text_embedding = store.find_all("TextEmbedding*")[0]
print(text_embedding('hi! What is your name?')[:10], '...')
# -> [0.0118862, -0.0006172658, -0.008183353, 0.02926386, -0.03759078, -0.031130238, -0.02367668 ...
