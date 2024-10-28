import os
from typing import List, Optional
from uuid import uuid4
import numpy as np
from pydantic import BaseModel, Field
from LLMAbstractModel.LLMsModel import LLMsStore, Model4LLMs
from LLMAbstractModel.utils import RegxExtractor

store = LLMsStore()
vendor = store.add_new_openai_vendor(api_key=os.environ.get('OPENAI_API_KEY','null'))
text_embedding = store.add_new_obj(Model4LLMs.TextEmbedding3Small())
llm = store.add_new_chatgpt4omini(vendor_id=vendor.get_id(),temperature=0.0)

# Node model for each node in the memory tree
class Node(BaseModel):
    id: str = Field(default_factory=lambda: f"Node:{uuid4()}")
    content: str = 'Root'
    embedding: list[float] = []
    parent_id: str = 'NULL'
    children: List['Node'] = Field(default_factory=list)
    depth: int = 0

    def swap(self, n:'Node'):
        self.content,n.content = n.content,self.content
        self.embedding,n.embedding = n.embedding,self.embedding

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

    def __init__(self,root=Node(),llm=None,text_embedding=None) -> None:
        self.llm:Model4LLMs.AbstractLLM=llm
        self.text_embedding:Model4LLMs.AbstractEmbedding=text_embedding
        if isinstance(root,dict): root = Node(**root)
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

    def classification(self, src: Node, ref: Node):
        sys = "## Instructions:¥nGiven two pieces of content, 'ref' and 'src', identify their relationship and label it with one of the following:¥nProvide only a single word as your output: Child, Parent, or Isolate.¥n¥n## Procedure:¥n1. Classify Each Content:¥n   - Determine whether 'ref' and 'src' are a topic or an action.¥n     - *Topic*: A general subject area.¥n     - *Action*: A specific activity within a topic.¥n¥n2. Assess Connection:¥n   - Evaluate whether there is any connection between 'ref' and 'src', considering both direct and indirect links.¥n¥n3. Apply Inclusion Rule:¥n   - Remember that a topic always includes its related actions.¥n¥n4. Compare Topics (if both are topics):¥n   - Determine if one topic includes the other.¥n¥n5. Select the Appropriate Label:¥n   - Choose Child if 'src' potentially includes 'ref'.¥n   - Choose Parent if 'ref' potentially includes 'src'.¥n   - Choose Isolate if no connection exists.¥n¥n## Notes:¥n- Carefully determine whether each piece of content is a topic or an action.¥n- Consider subtle connections for accurate labeling.¥n"
        self.llm.system_prompt=sys
        res:str = self.llm(f'## src¥n{src.content}¥n## ref¥n{ref.content}')
        return {'parent':-1,'isolate':0,'child':1}[res.strip().lower()]

    def get_threshold(self, depth: int) -> float:
        # Adaptive threshold calculation based on node depth
        return self.base_threshold * np.exp(self.lambda_factor * depth)
    
    def calc_embedding(self, node:Node):
        if len(node.embedding)>0:return node
        node.embedding = self.text_embedding(node.content)
        return node

    def similarity(self, src: Node, ref: Node)->float:
        # Cosine similarity calculation
        emb1, emb2 = self.calc_embedding(ref).embedding, self.calc_embedding(src).embedding
        emb1, emb2 = np.asarray(emb1), np.asarray(emb2)
        coss = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return coss
    
    def traverse(self, node:Node):
        # Yield the current node
        if node.content != 'Root':
            yield node
        # Recursively yield each child node
        for child in node.children:
            yield from self.traverse(child)

    def extract_all_embeddings(self):
        embeddings = {}
        for node in self.traverse(self.root):
            embeddings[node.id] = node.embedding
            node.embedding = []
        return embeddings
    
    def load_all_embeddings(self,embeddings:dict):
        for node in self.traverse(self.root):
            node.embedding = embeddings[node.id]
    
    def retrieve(self, query: str, top_k: int = 3) -> List[tuple[Node, float]]:
        nodes_with_scores = []
        query = self.calc_embedding(Node(content=query))

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
    
    def insert(self, content: str):
        # Use OpenAI's embedding function to get the embedding
        new_node = self.calc_embedding(Node(content=content))
        # Add to _node_dict
        MemTree.get_node(new_node.id)
        self._node_dict[new_node.id] = new_node
        self._insert_node(self.root, new_node)
        self.print_tree()

    def _insert_node(self, current_node: Node, new_node: Node):
        # Determine if we should add the new node as a child or traverse further
        best_clss = None
        best_child = None

        if len(current_node.children)==0:
            current_node.add_child(new_node)
            return

        for child in current_node.children:
            clss = self.classification(child, new_node)

            if clss<0:
                new_node.swap(child)
                self._insert_node(child, new_node)
                return

            if best_clss is None or clss >= best_clss:
                best_clss = clss
                best_child = child

        if clss==0:
            current_node.add_child(new_node)
            return
        
        if clss>0:
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

    def print_tree(self, node: Optional[Node] = None, level: int = 0):
        if node is None: node = self.root
        # Print current node with indentation corresponding to its depth
        print(" " * (level * 4) + f"- {node.content} (Depth: {node.depth})")
        for child in node.children:
            self.print_tree(child, level + 1)

# Sample complex personal data for Alex
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
memory_tree = MemTree(llm=llm,text_embedding=text_embedding)

# Insert each memory into the memory tree
for i in complex_sample_data: memory_tree.insert(i)

# Define a function to test retrieval with sample queries
def test_retrieval(query: str, top_k: int = 5):
    print(f"\nTop {top_k} matches for query: '{query}'")
    results = memory_tree.retrieve(query, top_k)
    for i, (node, score) in enumerate(results, start=1):
        print(f"{i}. score: {score:.4f} | [ {node.content} ]'")

# Testing with example queries that reflect personal scenarios
test_retrieval("Remind me about family events", top_k=6)
test_retrieval("Health and self-care routines", top_k=6)
test_retrieval("Work project deadlines", top_k=6)
test_retrieval("Weekend plans with friends", top_k=6)
test_retrieval("Personal development goals", top_k=6)

# Print the tree structure to visualize organization
print("\nMemory Tree Structure:")
memory_tree.print_tree()

# @descriptions('Workflow function of triage queries and routing to the appropriate agent', 
#               question='The question to ask the LLM')
class PersonalAssistantAgent(BaseModel):
    llm: Model4LLMs.AbstractLLM = llm
    text_embedding: Model4LLMs.AbstractEmbedding = text_embedding
    memory_root: Node = Node()
    memory_top_k: int = Field(default=5, ge=1, description="Number of top memories to retrieve.")
    system_prompt: str = "¥nYou are a capable, friendly assistant with a strong memory.¥nUse a mix of past information and new insights to answer effectively. ¥nFeel free to save something in memory without any permission, just reply with ```memory something```.¥nYou will be provided most relevant information.¥nSave new, important details like preferences or routines.¥n¥n## **Conversational Style**:¥n   - **Tone**: Be friendly, clear, and professional.¥n   - **Clarity and Conciseness**: Keep responses clear and to the point.¥n   - **Empathy and Politeness**: Show understanding, especially when users share concerns.¥n¥n## **Task Assistance**¥nAssist with information, suggestions, answering questions, and summaries as needed.¥n¥n## **Adaptability**:¥nAdjust to the user’s style, whether concise or detailed.¥n"

    def get_memory(self):
        return MemTree(self.memory_root,llm=self.llm,
                       text_embedding=self.text_embedding)
        
    def print_memory(self):
        self.get_memory().print_tree()

    def add_memory(self, content: str) -> None:
        self.get_memory().insert(content)

    def memory_retrieval(self, query: str) -> str:
        # Perform retrieval and format the results
        res = f"\nTop {self.memory_top_k} memories for query: '{query}'\n"
        results = self.get_memory().retrieve(query, self.memory_top_k)
        for i, (node, score) in enumerate(results, start=1):
            res += f"{i}. Score: {score:.3f} | Content: {node.content}\n"
        return res
    
    def __call__(self, query: str) -> str:
        memo_ext = RegxExtractor(regx=r"```memory\s*(.*)\s*```")
        # Retrieve relevant memory and format it for context
        memory = self.memory_retrieval(query)
        # Format the query with memory context
        query_with_context = f"{memory}\nUser Query: {query}"
        # Generate a response using the LLM
        self.llm.system_prompt=self.system_prompt
        response = self.llm(query_with_context)
        if '```memory' in response:
            new_memo = memo_ext(response)
            self.add_memory(new_memo)
        return response


agent = PersonalAssistantAgent(memory_root=memory_tree.root,llm=llm,text_embedding=text_embedding)
# print(agent('hi! Please tell me my events.'))
print(agent('Please remider me to schedule annual dental checkup in the first week of December, I am not decide the date yet.'))



