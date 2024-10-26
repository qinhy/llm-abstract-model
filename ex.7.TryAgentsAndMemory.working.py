import os
from typing import List, Optional
from uuid import uuid4
import numpy as np
from pydantic import BaseModel, Field
from LLMAbstractModel.LLMsModel import LLMsStore, Model4LLMs

store = LLMsStore()

vendor = store.add_new_openai_vendor(api_key=os.environ.get('OPENAI_API_KEY','null'))
text_embedding = store.add_new_obj(Model4LLMs.TextEmbedding3Small())
llm = store.add_new_chatgpt4omini(vendor_id=vendor.get_id(),temperature=0.0,system_prompt=
'''
## Instructions:

Given two pieces of content, 'ref' and 'src', identify their relationship and label it with one of the following:
Provide only a single word as your output: Child, Parent, or Isolate.


## Procedure:

1. Classify Each Content:
   - Determine whether 'ref' and 'src' are a topic or an action.
     - *Topic*: A general subject area.
     - *Action*: A specific activity within a topic.

2. Assess Connection:
   - Evaluate whether there is any connection between 'ref' and 'src', considering both direct and indirect links.

3. Apply Inclusion Rule:
   - Remember that a topic always includes its related actions.

4. Compare Topics (if both are topics):
   - Determine if one topic includes the other.

5. Select the Appropriate Label:
   - Choose Child if 'src' potentially includes 'ref'.
   - Choose Parent if 'ref' potentially includes 'src'.
   - Choose Isolate if no connection exists.


## Notes:

- Carefully determine whether each piece of content is a topic or an action.
- Consider subtle connections for accurate labeling.
''')


# Node model for each node in the memory tree
class Node(BaseModel):
    id: str = Field(default_factory=lambda: f"Node:{uuid4()}")
    content: str
    embedding: list[float] = []
    parent_id: str = 'NULL'
    children: List['Node'] = Field(default_factory=list)
    depth: int = 0

    def swap(self, n:'Node'):
        self.content,n.content = n.content,self.content
        self.embedding,n.embedding = n.embedding,self.embedding

    def calc_embedding(self):
        return self
        self.embedding = text_embedding(self.content)
        return self

    def add_child(self, child: 'Node'):
        self.children.append(child)
        child.parent_id = self.id
        child.depth = self.depth + 1
    
    def get_parent(self):
        # Retrieve parent node if it exists
        if self.parent_id:
            return MemTree.get_node(self.parent_id)
        return None

# MemTree model to manage the tree structure and memory operations
class MemTree:
    base_threshold: float = 0.4
    lambda_factor: float = 0.5
    _node_dict: dict[str, Node] = None

    def __init__(self,root = Node(content="Root", embedding=[], depth=0)) -> None:
        if isinstance(root,dict):
            root = Node(**root)
        self.root = root
        # Build _node_dict if _node_dict is None
        if MemTree._node_dict is None:
            MemTree._node_dict = {}
            def traverse(node: Node):
                MemTree._node_dict[node.id] = node
                for child in node.children:
                    traverse(child)
            # Traverse tree starting from root
            traverse(root)

    @staticmethod
    def get_node(id=''):
        return MemTree._node_dict.get(id, None)

    def similarity(self, src: Node, ref: Node):
        # Use OpenAI's embedding function to get the embedding for the ref
        ref = Node(content=content).calc_embedding()
        # Cosine similarity calculation
        # emb1, emb2 = src.embedding, ref.embedding
        # emb1, emb2 = np.asarray(emb1), np.asarray(emb2)
        # coss = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        res:str = llm(f'## src¥n{src.content}¥n## ref¥n{ref.content}')
        # print(res,', ',src.content,', ',ref.content)

        return {'parent':-1,'isolate':0,'child':1}[res.strip().lower()]

    def get_threshold(self, depth: int) -> float:
        # Adaptive threshold calculation based on node depth
        return self.base_threshold * np.exp(self.lambda_factor * depth)
    
    def insert(self, content: str):
        # Use OpenAI's embedding function to get the embedding
        new_node = Node(content=content).calc_embedding()
        
        # Add to _node_dict
        MemTree.get_node(new_node.id)
        self._node_dict[new_node.id] = new_node
        
        self._insert_node(self.root, new_node)

        self.print_tree()

    def _insert_node(self, current_node: Node, new_node: Node):
        # Determine if we should add the new node as a child or traverse further
        best_similarity = -1
        best_child = None

        if len(current_node.children)==0:
            current_node.add_child(new_node)
            return

        for child in current_node.children:
            sim = self.similarity(child, new_node)

            if sim<0:
                new_node.swap(child)
                self._insert_node(child, new_node)

            if sim >= best_similarity:
                best_similarity = sim
                best_child = child

        if sim==0:
            current_node.add_child(new_node)
            return
        
        if sim>0:
            self._insert_node(best_child, new_node)
            return
        
        # if best_child and best_similarity > 0:#self.get_threshold(current_node.depth):
        #     # Traverse further if similarity threshold is met
        #     self._insert_node(best_child, new_node)
        # else:
        #     # Otherwise, add new_node as a child of current_node
        #     current_node.add_child(new_node)
        #     new_node.parent_id = current_node.id
        #     new_node.depth = current_node.depth + 1

    def retrieve(self, query: str, top_k: int = 3) -> List[tuple[Node, float]]:
        nodes_with_scores = []

        def traverse(node: Node):
            if node.content != 'Root':
                sim = self.similarity(node, query)
                nodes_with_scores.append((node, sim))
            for child in node.children:
                traverse(child)

        # Traverse tree starting from root
        traverse(self.root)

        # Sort by similarity and return top_k results
        nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [(node, score) for node, score in nodes_with_scores[:top_k]]

    def print_tree(self, node: Optional[Node] = None, level: int = 0):
        """Prints the tree structure with indentation based on node depth."""
        if node is None:
            node = self.root

        # Print current node with indentation corresponding to its depth
        print(" " * (level * 4) + f"- {node.content} (Depth: {node.depth})")
        for child in node.children:
            self.print_tree(child, level + 1)


# Sample complex personal data for Alex


complex_sample_data = [
    "The cat sat on the mat",
    "Dogs are friendly animals",
    "Cats are independent creatures",
    "Birds can fly",
    "Fish live in water",
    # "The dog chased the cat",
    # "Cats and dogs are popular pets",
    # "Birds build nests in trees",
    # "Fish have gills for breathing",
    # "The mat is a comfortable place for the cat",
]

complex_sample_data = [
    '''Work-related''',
    "Work on quarterly report on sales trends by end of September",
    "Team meeting every Monday at 10 AM to discuss project updates",
    # "Read the latest research paper on artificial intelligence in medicine",
    # "Finish the product pitch presentation for the client next week",
    # "Coordinate with the marketing team on the new campaign for social media",
    
    '''Personal relationships and social nets''',
    "Birthday dinner for Emily on October 12th at her favorite Italian restaurant",
    "Remind Mom about her doctor's appointment next Tuesday",
    # "Catch up with John over coffee this weekend to discuss the hiking trip",
    # "Send anniversary wishes to Sarah and Mike on November 5th",
    # "Plan a game night with friends for Friday evening",
    
    '''Hobbies and personal interests''',
    "Practice guitar chords for 'Hey Jude' by The Beatles",
    "Sign up for a pottery class to explore new creative outlets",
    # "Research the latest DSLR cameras for landscape photography",
    # "Look into joining a weekend hiking club for outdoor activities",
    # "Try a new recipe for Thai curry with coconut milk this weekend",
    
    '''Health and self-care''',
    "Yoga session every morning at 6:30 AM for better flexibility",
    "Drink more water throughout the day to stay hydrated",
    # "Schedule annual dental checkup in the first week of December",
    # "Take vitamin supplements daily for general well-being",
    # "Set a reminder to take a 5-minute break every hour when working",
    
    '''Goals and personal development''',
    "Complete an online course on data science by the end of this month",
    "Read one new book each month; currently reading 'Sapiens' by Yuval Noah Harari",
    # "Practice mindfulness meditation in the evening to reduce stress",
    # "Write a journal entry every night to reflect on daily events",
    # "Set a target to run 5 kilometers without stopping by the end of the year",
    
    '''Random thoughts and reflections''',
    "Consider adopting a pet dog; research breeds that are good with kids",
    "The sunset at the beach yesterday was beautiful, and I’d love to visit again soon",
    # "Wondering if switching to a standing desk would help with posture",
    # "Had a deep conversation with Sarah about life goals and future plans",
    # "Noticed that productivity peaks after a good night's sleep",
    
    '''reminders and notes''',
    "Buy groceries: milk, eggs, bread, and fresh vegetables",
    "Research flight options for the vacation trip in December",
    # "Pick up the dry cleaning by Thursday evening",
    # "Renew library membership by the end of the month",
    # "Replace the batteries in the smoke detector this weekend",
]

# Example usage of MemTree with complex personal data
memory_tree = MemTree()

# Insert each memory into the memory tree
for content in complex_sample_data:
    memory_tree.insert(content)

# Define a function to test retrieval with sample queries
# def test_retrieval(query: str, top_k: int = 5):
#     print(f"\nTop {top_k} matches for query: '{query}'")
#     results = memory_tree.retrieve(query, top_k)
#     for i, (node, score) in enumerate(results, start=1):
#         print(f"{i}. Content: '{node.content}' | Similarity: {score:.4f}")

# # Testing with example queries that reflect personal scenarios
# test_retrieval("Remind me about family events", top_k=3)
# test_retrieval("Health and self-care routines", top_k=3)
# test_retrieval("Work project deadlines", top_k=3)
# test_retrieval("Weekend plans with friends", top_k=3)
# test_retrieval("Personal development goals", top_k=3)

# Print the tree structure to visualize organization
print("\nMemory Tree Structure:")
memory_tree.print_tree()


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
