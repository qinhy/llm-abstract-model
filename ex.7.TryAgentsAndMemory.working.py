import os
from typing import List, Optional
import numpy as np
import requests
from pydantic import BaseModel, Field

# OpenAI API function to get embeddings
def get_openai_embedding(content: str, api_key: str=os.getenv('OPENAI_API_KEY'), model: str = "text-embedding-3-small") -> list[float]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": content,
        "model": model
    }
    response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=data)
    response.raise_for_status()  # Raise an error for bad responses
    embedding = response.json()["data"][0]["embedding"]
    return embedding

# Node model for each node in the memory tree
class Node(BaseModel):
    content: str
    embedding: list[float] = None
    parent: Optional['Node'] = None
    children: List['Node'] = Field(default_factory=list)
    depth: int = 0

    def add_child(self, child: 'Node'):
        self.children.append(child)

# MemTree model to manage the tree structure and memory operations
class MemTree(BaseModel):
    root: Node = Field(default_factory=lambda: Node(content="Root", embedding=[], depth=0))
    base_threshold: float = 0.4
    lambda_factor: float = 0.5

    def similarity(self, emb1: list[float], emb2: list[float]) -> float:
        # Cosine similarity calculation
        emb1, emb2 = np.asarray(emb1), np.asarray(emb2)
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def get_threshold(self, depth: int) -> float:
        # Adaptive threshold calculation based on node depth
        return self.base_threshold * np.exp(self.lambda_factor * depth)

    def get_embedding(self, content: str):
        return get_openai_embedding(content)
    
    def insert(self, content: str):
        # Use OpenAI's embedding function to get the embedding
        embedding = self.get_embedding(content)
        new_node = Node(content=content, embedding=embedding)
        self._insert_node(self.root, new_node)

    def _insert_node(self, current_node: Node, new_node: Node):
        # Determine if we should add the new node as a child or traverse further
        best_similarity = -1
        best_child = None

        for child in current_node.children:
            sim = self.similarity(child.embedding, new_node.embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_child = child

        # Check if best similarity meets threshold
        if best_child and best_similarity >= self.get_threshold(current_node.depth):
            # Traverse further if similarity threshold is met
            self._insert_node(best_child, new_node)
        else:
            # Otherwise, add new_node as a child of current_node
            current_node.add_child(new_node)
            new_node.parent = current_node
            new_node.depth = current_node.depth + 1

    def retrieve(self, query: str, top_k: int = 3) -> List[Node]:
        # Use OpenAI's embedding function to get the embedding for the query
        query_embedding = self.get_embedding(query)
        nodes_with_scores = []

        def traverse(node: Node):
            if node.content!='Root':
                sim = self.similarity(node.embedding, query_embedding)
                nodes_with_scores.append((node, sim))
            for child in node.children:
                traverse(child)

        # Traverse tree starting from root
        traverse(self.root)

        # Sort by similarity and return top_k results
        nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [node for node, score in nodes_with_scores[:top_k]]

    def print_tree(self, node: Optional[Node] = None, level: int = 0):
        """Prints the tree structure with indentation based on node depth."""
        if node is None:
            node = self.root

        # Print current node with indentation corresponding to its depth
        print(" " * (level * 4) + f"- {node.content} (Depth: {node.depth})")
        for child in node.children:
            self.print_tree(child, level + 1)


# Sample dataset of content strings with manually assigned 4-dimensional embeddings
sample_data = [
    "The cat sat on the mat",
    "Dogs are friendly animals",
    "Cats are independent creatures",
    "Birds can fly",
    "Fish live in water",
    "The dog chased the cat",
    "Cats and dogs are popular pets",
    "Birds build nests in trees",
    "Fish have gills for breathing",
    "The mat is a comfortable place for the cat",
]

# Display the dataset to verify the embeddings
for entry in sample_data:
    print(f"Content: {entry}\n")

# Initialize the tree and insert nodes from the sample dataset
tree = MemTree()
for entry in sample_data:
    tree.insert(entry)
tree.print_tree()

# Retrieve similar nodes to a query
query = "Cats are independent creatures"
top_k_results = tree.retrieve(query=query, top_k=3)

# Display the top-k retrieved nodes
print("\nTop-k retrieved nodes based on similarity to the query:")
for i, node in enumerate(top_k_results, start=1):
    print(f"{i}. Content: {node.content}")
    

# import requests
# from LLMAbstractModel.utils import RegxExtractor
# from LLMAbstractModel import LLMsStore,Model4LLMs
# descriptions = Model4LLMs.Function.param_descriptions
# def myprint(string):
#     print('##',string,':\n',eval(string),'\n')

# store = LLMsStore()
# vendor = store.add_new_openai_vendor(api_key="OPENAI_API_KEY")
# debug = True

# import requests
# import os
# import json

# @descriptions(
#     'This function retrieves the most relevant memories from a vector memory store based on a given question. '
#     'It uses vector embeddings to find and rank the top N similar memories in the store.',
#     question='A string input representing the query or question for which relevant memories need to be retrieved.'
# )
# class SimpleLLMMemory(Model4LLMs.Function):
#     top_n: int = 10

#     def __call__(self, question: str):
#         # Convert question to vector using OpenAI API
#         question_vec = self.vectorize_question(question)
        
#         # Search memory vector store for top_n most similar vectors
#         memories = self.search_memory(question_vec, top_n=self.top_n)
        
#         # Sort memories by similarity score (descending order)
#         sorted_memories = sorted(memories, key=lambda x: x['similarity'], reverse=True)
        
#         # Return only the most relevant memory data
#         top_memories = [memory['data'] for memory in sorted_memories]
        
#         return top_memories

#     def vectorize_question(self, question: str):
#         # Set up the API URL
#         url = "https://api.openai.com/v1/embeddings"

#         # Set up headers including the OpenAI API key
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
#         }

#         # Prepare the data payload for the POST request
#         data = {
#             "model": "text-embedding-ada-002",  # The embedding model from OpenAI
#             "input": question
#         }

#         # Make the API request to OpenAI
#         response = requests.post(url, headers=headers, data=json.dumps(data))

#         # Check if the request was successful
#         if response.status_code != 200:
#             raise Exception(f"Error: {response.status_code}, {response.text}")

#         # Extract the embedding vector from the response
#         embedding = response.json()['data'][0]['embedding']

#         return embedding

#     def search_memory(self, question_vec, top_n: int):
#         # Placeholder for searching vector memory store
#         # This would typically involve something like cosine similarity or a nearest neighbor search
#         return memory_store.search_by_vector(question_vec, top_n=top_n)

# # Make sure to set up your OpenAI API key before using the OpenAI API
# # Example: export OPENAI_API_KEY="your-api-key" in the environment

# # Add functions for reverse geocoding and address extraction
# french_address_search_function = store.add_new_obj(FrenchReverseGeocodeFunction())
