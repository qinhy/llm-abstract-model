import datetime
import json
import os
from typing import List, Optional
from uuid import uuid4
import numpy as np
from pydantic import BaseModel, Field
from LLMAbstractModel.LLMsModel import LLMsStore, Model4LLMs
from LLMAbstractModel.utils import RegxExtractor

# Node model for each node in the memory treefrom typing import List, Optional
class TextContentNode(BaseModel):
    id: str = Field(default_factory=lambda: f"Node:{uuid4()}")
    content: str = "Root(Group)"
    embedding: List[float] = Field(default_factory=list)

    parent_id: str = "NULL"
    children: List['TextContentNode'] = Field(default_factory=list)
    depth: int = 0

    _root_node: 'TextContentNode' = None

    def reflesh_root(self):
        if self.is_root():
            for child in self.traverse(self):
                if not child.is_root():
                    child._root_node = self

    def traverse(self, node: 'TextContentNode'):
        """Yield each node in the tree starting from a given node."""
        yield node
        for child in node.children:
            yield from self.traverse(child)

    def get_node(self, node_id: str) -> Optional['TextContentNode']:
        """Retrieve a node by search ID recursively within children."""
        root = self if self.is_root() else self._root_node        
        for child in root.traverse(root):
            if child.id == node_id:
                return child
        # # If no node found, return None
        return None

    def is_group(self) -> bool:
        """Check if this node represents a group."""
        return "Group" in self.content

    def is_root(self) -> bool:
        """Check if this node is the root."""
        return self.content == "Root(Group)"
    
    def clone(self) -> 'TextContentNode':
        """Create a deep copy of this node and all its descendants."""
        new_node = TextContentNode(
            content=self.content,
            embedding=self.embedding[:],
            depth=self.depth
        )
        for child in self.children:
            cloned_child = child.clone()  # Recursively clone each child
            new_node.add(cloned_child)  # Add cloned child to the new node
        return new_node
    
    def dispose(self):
        """Remove this node from its parent's children list."""
        parent = self.get_parent()
        if parent:
            parent.children = [child for child in parent.children if child.id != self.id]

    def swap(self, other: 'TextContentNode'):
        """Swap content and embeddings with another node."""
        self.content, other.content = other.content, self.content
        self.embedding, other.embedding = other.embedding, self.embedding

    def add(self, child: 'TextContentNode'):
        """Add a child to this node."""
        self.children.append(child)
        child.parent_id = self.id
        child.depth = self.depth + 1

    def get_parent(self) -> Optional['TextContentNode']:
        """Retrieve the parent node from the tree."""
        return self.get_node(self.parent_id) if self.parent_id else None

    def groups(self) -> List['TextContentNode']:
        """Return a list of groups along the path to the root."""
        return [TextContentNode(content=p) for p in self.get_path_to_root() if TextContentNode(content=p).is_group()]

    def get_all_children(self) -> List['TextContentNode']:
        """Retrieve all descendants of this node."""
        descendants = []
        def collect_children(node: 'TextContentNode'):
            descendants.append(node)
            for child in node.children:
                collect_children(child)
        collect_children(self)
        return descendants

    def move_to(self, new_parent: 'TextContentNode'):
        """Move this node under a new parent node and update depth for all descendants."""
        if self.get_parent():self.dispose()
        new_parent.add(self)
        old_depth = self.depth
        self.depth = new_parent.depth + 1
        for child in self.get_all_children():
            child.depth = child.depth - old_depth + self.depth

    def get_path_to_root(self) -> List[str]:
        """Retrieve the path from this node to the root as a list of contents."""
        path = []
        current = self
        while current and not current.is_root():
            path.append(current.content)
            current = current.get_parent()
        return list(reversed(path))[:-1]

    def has_keyword(self, keyword: str) -> bool:
        """Check if the node's content or any descendant's content contains the keyword."""
        if keyword in self.content:
            return True
        return any(child.has_keyword(keyword) for child in self.children)

class TextMemoryTree:
    base_threshold: float = 0.4
    lambda_factor: float = 0.5
    _embedding_cache_dict: dict[str, TextContentNode] = {}

    def __init__(self, root: TextContentNode = None, llm=None, text_embedding=None):
        self.llm:Model4LLMs.AbstractLLM=llm
        self.text_embedding:Model4LLMs.AbstractEmbedding=text_embedding
        if isinstance(root,dict): root = TextContentNode(**root)
        self.root = root
        self.root.reflesh_root()
        # Build _node_dict if _node_dict is None
        for node in self.traverse(self.root):
            TextMemoryTree._embedding_cache_dict[node.content] = node.embedding

    def traverse(self, node: TextContentNode):
        """Yield each node in the tree starting from a given node."""
        yield node
        for child in node.children:
            yield from self.traverse(child)

    def get_node(self, node_id: str) -> Optional[TextContentNode]:
        """Retrieve a node by its ID."""
        return self.root.get_node(node_id)

    def remove_content(self, content: str):
        """Remove all nodes that contain specific content."""
        to_remove = [node for node in self.traverse(self.root) if node.content == content]
        for node in to_remove:
            node.dispose()

    def find_nodes_by_content(self, keyword: str) -> List[TextContentNode]:
        """Find all nodes containing a specific keyword in their content."""
        return [node for node in self.traverse(self.root) if keyword in node.content]

    def get_leaf_nodes(self) -> List[TextContentNode]:
        """Retrieve all leaf nodes (nodes without children)."""
        return [node for node in self.traverse(self.root) if not node.children]

    def reparent_node(self, node_id: str, new_parent_id: str):
        """Reparent a node under a new parent."""
        node = self.get_node(node_id)
        new_parent = self.get_node(new_parent_id)
        if node and new_parent:
            node.move_to(new_parent)

    def classification(self, src: TextContentNode, ref: TextContentNode):
        sys = "## Instructions:¥nGiven two pieces of content, 'ref' and 'src', identify their relationship and label it with one of the following:¥nProvide only a single word as your output: Child, Parent, or Isolate.¥n¥n## Procedure:¥n1. Classify Each Content:¥n   - Determine whether 'ref' and 'src' are a topic or an action.¥n     - *Topic*: A general subject area.¥n     - *Action*: A specific activity within a topic.¥n¥n2. Assess Connection:¥n   - Evaluate whether there is any connection between 'ref' and 'src', considering both direct and indirect links.¥n¥n3. Apply Inclusion Rule:¥n   - Remember that a topic always includes its related actions.¥n¥n4. Compare Topics (if both are topics):¥n   - Determine if one topic includes the other.¥n¥n5. Select the Appropriate Label:¥n   - Choose Child if 'src' potentially includes 'ref'.¥n   - Choose Parent if 'ref' potentially includes 'src'.¥n   - Choose Isolate if no connection exists.¥n¥n## Notes:¥n- Carefully determine whether each piece of content is a topic or an action.¥n- Consider subtle connections for accurate labeling.¥n"
        self.llm.system_prompt=sys
        res:str = self.llm(f'## src¥n{src.content}¥n## ref¥n{ref.content}')
        return {'parent':-1,'isolate':0,'child':1}[res.strip().lower()]

    def tidytree(self):
        sys = '''Please organize the following memory list and respond in the original tree structure format. If a node represents a group, add 'Group' at the end of the node name. Feel free to add new group nodes as needed.'''
        self.llm.system_prompt=sys
        res:str = self.llm(f'```text\n{self.print_tree()}\n```')
        res:str = RegxExtractor(regx=r"```text\s*(.*)\s*```")(res)
        print(res)
        root = self.parse_text_tree(res)
        embeddings = {}
        for node in self.traverse(self.root):
            embeddings[node.content] = node.embedding
            node.embedding = []
        
        for node in self.traverse(root):
            node.embedding = embeddings.get(node.content,[])
            
        self.root = root
        return self
        
    def get_threshold(self, depth: int) -> float:
        # Adaptive threshold calculation based on node depth
        return self.base_threshold * np.exp(self.lambda_factor * depth)
    
    def calc_embedding(self, node:TextContentNode):
        if len(node.embedding)>0:return node
        if node.is_group():
            node.embedding = np.zeros(self.text_embedding.embedding_dim).tolist()
        else:
            if node.content in TextMemoryTree._embedding_cache_dict:
                node.embedding = TextMemoryTree._embedding_cache_dict[node.content]
            if len(node.embedding)==0:
                node.embedding = self.text_embedding(node.content)
        return node

    def similarity(self, src: TextContentNode, ref: TextContentNode)->float:
        # Cosine similarity calculation
        emb1, emb2 = self.calc_embedding(ref).embedding, self.calc_embedding(src).embedding
        emb1, emb2 = np.asarray(emb1), np.asarray(emb2)
        norm = (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        if norm == 0:return 0.0
        coss = np.dot(emb1, emb2) / norm
        return coss
    
    def extract_all_embeddings(self):
        embeddings = {}
        for node in self.traverse(self.root):
            embeddings[node.id] = node.embedding
            node.embedding = []
        return embeddings
    
    def dump_all_embeddings(self,path='embeddings.json'):
        embeddings = self.extract_all_embeddings()
        with open(path, "w") as tf: tf.write(json.dumps(embeddings))
    
    def put_all_embeddings(self,embeddings:dict):
        for node in self.traverse(self.root):
            node.embedding = embeddings[node.id]
    
    def load_all_embeddings(self,path='embeddings.json'):
        with open(path, "r") as tf: self.put_all_embeddings(json.loads(tf.read()))
        return self
    
    def retrieve(self, query: str, top_k: int = 3) -> List[tuple[TextContentNode, float]]:
        nodes_with_scores = []
        query = self.calc_embedding(TextContentNode(content=query))

        for node in self.traverse(self.root):
            if node.is_root():continue
            sim = self.similarity(node, query)
            nodes_with_scores.append((node, sim))

        # Sort by similarity and return top_k results
        nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [(node, score) for node, score in nodes_with_scores[:top_k]]
    
    def insert(self, content: str):
        # Use OpenAI's embedding function to get the embedding
        new_node = self.calc_embedding(TextContentNode(content=content.strip()))
        self._insert_node(self.root, new_node)
        self.print_tree()
        self.root.reflesh_root()

    def _insert_node(self, current_node: TextContentNode, new_node: TextContentNode):
        # Determine if we should add the new node as a child or traverse further
        # best_clss = None
        # best_child = None

        # if len(current_node.children)==0:
        #     current_node.add_child(new_node)
        #     return

        # for child in current_node.children:
        #     clss = self.classification(child, new_node)

        #     if clss<0:
        #         new_node.swap(child)
        #         self._insert_node(child, new_node)
        #         return

        #     if best_clss is None or clss >= best_clss:
        #         best_clss = clss
        #         best_child = child

        # if clss==0:
        #     current_node.add_child(new_node)
        #     return
        
        # if clss>0:
        #     self._insert_node(best_child, new_node)
        #     return
        
        
        # Determine if we should add the new node as a child or traverse further
        best_similarity = -1
        best_child = None

        for child in current_node.children:
            sim = self.similarity(child, new_node)
            if sim > best_similarity:
                best_similarity = sim
                best_child = child

        # Check if best similarity meets threshold
        if best_child and best_similarity >= self.get_threshold(current_node.depth):
            # Traverse further if similarity threshold is met
            self._insert_node(best_child, new_node)
        else:
            # Otherwise, add new_node as a child of current_node
            current_node.add(new_node)

    def parse_text_tree(self, text:str):
        lines = text.splitlines()
        # Skip empty lines
        lines = [line for line in lines if line.lstrip()]
        
        stack = [(TextContentNode(), -1)]  # Stack with root item
        root = stack[0][0]

        if 'Root' not in lines[0]:raise ValueError('first Node must be Root.')
        lines.pop(0)

        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            content = stripped[2:].strip()  # Remove bullet and any additional whitespace
            level = indent // 4  # Determine level based on indent (4 spaces per level)
            
            # Pop stack to match current indentation level
            while stack and stack[-1][1] >= level:
                stack.pop()
            
            # Append new item to current level's parent
            parent, _ = stack[-1]
            item = TextContentNode(content=content)
            parent.add(item)
            
            # Prepare to add child items in subsequent lines
            stack.append((item, level))
        
        root.reflesh_root()
        return root
    
    def print_tree(self, node: Optional[TextContentNode] = None, level: int = 0):
        res = ''
        if node is None: node = self.root
        # Print current node with indentation corresponding to its depth
        res += " " * (level * 4) + f"- {node.content.strip()}\n"
        for child in node.children:
            res += self.print_tree(child, level + 1)
        if node == self.root:
            print(res[:-1])
            res = res[:-1]
        return res
    
def test_tree():
    def sample_tree():
        # Create a sample tree with root, two children, and a grandchild for testing
        root = TextContentNode(content="Root(Group)")
        child1 = TextContentNode(content="Child 1")
        child2 = TextContentNode(content="Child 2")
        grandchild1 = TextContentNode(content="Grandchild 1")
        
        # Build tree structure
        root.add(child1)
        root.add(child2)
        child1.add(grandchild1)
        
        tree = TextMemoryTree(root=root)
        return tree,[root, child1, child2, grandchild1]

    def test_add_node(sample_tree:tuple[TextMemoryTree,list[TextContentNode]]):
        tree, [root, child1, child2, grandchild1] = sample_tree
        assert len(root.children) == 2
        assert child1 in root.children
        assert child2 in root.children
        assert grandchild1 in child1.children

    def test_dispose_node(sample_tree:tuple[TextMemoryTree,list[TextContentNode]]):
        tree, [root, child1, child2, grandchild1] = sample_tree
        child1.dispose()
        assert len(root.children) == 1
        assert child1 not in root.children

    def test_swap_nodes(sample_tree:tuple[TextMemoryTree,list[TextContentNode]]):
        tree, [root, child1, child2, grandchild1] = sample_tree
        content_before = child1.content
        child1.swap(child2)
        assert child2.content == content_before

    def test_move_to(sample_tree:tuple[TextMemoryTree,list[TextContentNode]]):
        tree, [root, child1, child2, grandchild1] = sample_tree
        grandchild1.move_to(child2)
        assert grandchild1 in child2.children
        assert grandchild1 not in child1.children
        assert grandchild1.depth == child2.depth + 1

    def test_get_parent(sample_tree:tuple[TextMemoryTree,list[TextContentNode]]):
        tree, [root, child1, child2, grandchild1] = sample_tree
        assert child1.get_parent() == root
        assert grandchild1.get_parent() == child1

    def test_get_all_children(sample_tree:tuple[TextMemoryTree,list[TextContentNode]]):
        tree, [root, child1, child2, grandchild1] = sample_tree
        descendants = child1.get_all_children()
        assert grandchild1 in descendants
        assert child2 not in descendants

    def test_get_path_to_root(sample_tree:tuple[TextMemoryTree,list[TextContentNode]]):
        tree, [root, child1, child2, grandchild1] = sample_tree
        path = grandchild1.get_path_to_root()
        assert path == ["Child 1"]

    def test_is_root(sample_tree:tuple[TextMemoryTree,list[TextContentNode]]):
        tree, [root, child1, child2, grandchild1] = sample_tree
        assert root.is_root() is True
        assert child1.is_root() is False

    def test_is_group(sample_tree:tuple[TextMemoryTree,list[TextContentNode]]):
        tree, [root, child1, child2, grandchild1] = sample_tree
        assert root.is_group() is True
        assert child1.is_group() is False

    def test_has_keyword(sample_tree:tuple[TextMemoryTree,list[TextContentNode]]):
        tree, [root, child1, child2, grandchild1] = sample_tree
        assert root.has_keyword("Root") is True
        assert grandchild1.has_keyword("Nonexistent") is False

    def test_find_nodes_by_content(sample_tree:tuple[TextMemoryTree,list[TextContentNode]]):
        tree, [root, child1, child2, grandchild1] = sample_tree
        found_nodes = tree.find_nodes_by_content("Child")
        assert child1 in found_nodes
        assert child2 in found_nodes
        assert grandchild1 not in found_nodes

    def test_remove_content(sample_tree:tuple[TextMemoryTree,list[TextContentNode]]):
        tree, [root, child1, child2, grandchild1] = sample_tree
        tree.remove_content("Child 1")
        assert child1 not in root.children

    def test_get_leaf_nodes(sample_tree:tuple[TextMemoryTree,list[TextContentNode]]):
        tree, [root, child1, child2, grandchild1] = sample_tree
        leaf_nodes = tree.get_leaf_nodes()
        assert grandchild1 in leaf_nodes
        assert child1 not in leaf_nodes  # since it has a child

    def test_clone(sample_tree:tuple[TextMemoryTree,list[TextContentNode]]):
        tree, [root, child1, child2, grandchild1] = sample_tree
        subclone = child1.clone()
        assert subclone is not None
        assert subclone.content == child1.content
        assert len(subclone.children) == len(child1.children)
        assert subclone.children[0].content == grandchild1.content

    def test_reparent_node(sample_tree:tuple[TextMemoryTree,list[TextContentNode]]):
        tree, [root, child1, child2, grandchild1] = sample_tree
        tree.reparent_node(grandchild1.id, child2.id)
        assert grandchild1 in child2.children
        assert grandchild1 not in child1.children
        assert grandchild1.depth == child2.depth + 1


    test_add_node(sample_tree())
    test_dispose_node(sample_tree())
    test_swap_nodes(sample_tree())
    test_move_to(sample_tree())
    test_get_parent(sample_tree())
    test_get_all_children(sample_tree())
    test_get_path_to_root(sample_tree())
    test_is_root(sample_tree())
    test_is_group(sample_tree())
    test_has_keyword(sample_tree())
    test_find_nodes_by_content(sample_tree())
    test_remove_content(sample_tree())
    test_get_leaf_nodes(sample_tree())
    test_clone(sample_tree())
    test_reparent_node(sample_tree())

class PersonalAssistantAgent(BaseModel):
    llm: Model4LLMs.AbstractLLM = None
    text_embedding: Model4LLMs.AbstractEmbedding = None
    memory_root: TextContentNode = TextContentNode()
    memory_top_k: int = Field(default=5, ge=1, description="Number of top memories to retrieve.")
    system_prompt: str = "You are a capable, friendly assistant use a mix of past information and new insights to answer effectively. Save new, important details—such as preferences or routines—in your own words. Simply reply with ```memory ... ```."

    def tidy_memory(self):
        self.memory_root = TextMemoryTree(self.memory_root,llm=self.llm,
                       text_embedding=self.text_embedding).tidytree().root

    def get_memory(self):
        return TextMemoryTree(self.memory_root,llm=self.llm,
                       text_embedding=self.text_embedding)
        
    def print_memory(self):
        self.get_memory().print_tree()

    def add_memory(self, content: str) -> None:
        self.get_memory().insert(content)

    def memory_retrieval(self, query: str) -> str:
        # Perform retrieval and format the results
        res = f"## Top {self.memory_top_k} memories for the query:\n"
        results = self.get_memory().retrieve(query, self.memory_top_k)
        if len(results)==0:return res+"No memories.\n"
        for i, (node, score) in enumerate(results, start=1):
            res += f"{i}. Score: {score:.3f} | Content: {node.content}\n"
        return res
    
    def __call__(self, query: str, print_memory=True) -> str:
        query = f"## User Query\n{query} ({str(datetime.datetime.now())})\n"
        memo_ext = RegxExtractor(regx=r"```memory\s*(.*)\s*```")
        # Retrieve relevant memory and format it for context
        memory = self.memory_retrieval(query)
        # Format the query with memory context
        query = f"{memory}\n{query}"
        if print_memory:
            print("############ For Debug ##############")
            print(query)
            print("#####################################")
            print()

        # Generate a response using the LLM
        self.llm.system_prompt=self.system_prompt
        response = self.llm(query)
        if '```memory' in response:
            new_memo = memo_ext(response)
            self.add_memory(new_memo)
        return response


# main usage form here
# Sample complex personal data for Alex
complex_sample_data = [
    '''Work-related''',
    "Work on quarterly report on sales trends by end of September",
    "Team meeting every Monday at 10 AM to discuss project updates",
    "Read the latest research paper on artificial intelligence in medicine",
    "Finish the product pitch presentation for the client next week",
    "Coordinate with the marketing team on the new campaign for social media",
    
    '''Personal relationships and social nets''',
    "Birthday dinner for Emily on October 12th at her favorite Italian restaurant",
    "Remind Mom about her doctor's appointment next Tuesday",
    "Catch up with John over coffee this weekend to discuss the hiking trip",
    "Send anniversary wishes to Sarah and Mike on November 5th",
    "Plan a game night with friends for Friday evening",
    
    '''Hobbies and personal interests''',
    "Practice guitar chords for 'Hey Jude' by The Beatles",
    "Sign up for a pottery class to explore new creative outlets",
    "Research the latest DSLR cameras for landscape photography",
    "Look into joining a weekend hiking club for outdoor activities",
    "Try a new recipe for Thai curry with coconut milk this weekend",
    
    '''Health and self-care''',
    "Yoga session every morning at 6:30 AM for better flexibility",
    "Drink more water throughout the day to stay hydrated",
    "Take vitamin supplements daily for general well-being",
    "Set a reminder to take a 5-minute break every hour when working",
    
    '''Goals and personal development''',
    "Complete an online course on data science by the end of this month",
    "Read one new book each month; currently reading 'Sapiens' by Yuval Noah Harari",
    "Practice mindfulness meditation in the evening to reduce stress",
    "Write a journal entry every night to reflect on daily events",
    "Set a target to run 5 kilometers without stopping by the end of the year",
    
    '''Random thoughts and reflections''',
    "Consider adopting a pet dog; research breeds that are good with kids",
    "The sunset at the beach yesterday was beautiful, and I’d love to visit again soon",
    "Wondering if switching to a standing desk would help with posture",
    "Had a deep conversation with Sarah about life goals and future plans",
    "Noticed that productivity peaks after a good night's sleep",
    
    '''reminders and notes''',
    "Buy groceries: milk, eggs, bread, and fresh vegetables",
    "Research flight options for the vacation trip in December",
    "Pick up the dry cleaning by Thursday evening",
    "Renew library membership by the end of the month",
    "Replace the batteries in the smoke detector this weekend",
]


# Initialize the LLM Store and vendor
store = LLMsStore()
vendor = store.add_new_openai_vendor(api_key=os.environ.get('OPENAI_API_KEY', 'null'))

# Add the necessary components
text_embedding = store.add_new_obj(Model4LLMs.TextEmbedding3Small())
llm = store.add_new_chatgpt4omini(vendor_id=vendor.get_id(), temperature=0.0)

# Create the root node and initialize the memory tree
root = TextContentNode()
memory_tree = TextMemoryTree(root, llm=llm, text_embedding=text_embedding)

# Insert complex sample data into the memory tree
for memory in complex_sample_data:
    memory_tree.insert(memory)

# Print the tree structure to visualize the organization
print("\nMemory Tree Structure:")
memory_tree.print_tree()

print("\nTidied Memory Tree Structure:")
memory_tree.tidytree()

# Define a function to test memory retrieval with various sample queries
def test_retrieval(memory_tree: TextMemoryTree, query: str, top_k: int = 5):
    print(f"\nTop {top_k} matches for query: '{query}'")
    results = memory_tree.retrieve(query, top_k)
    for i, (node, score) in enumerate(results, start=1):
        print(f"{i}. score: {score:.4f} | [ {node.content} ]")

# Sample queries reflecting various personal scenarios
queries = [
    "Remind me about family events",
    "Health and self-care routines",
    "Work project deadlines",
    "Weekend plans with friends",
    "Personal development goals"
]

# Run retrieval tests
for query in queries:
    test_retrieval(memory_tree, query, top_k=6)

# Initialize the personal assistant agent using the memory tree
agent = PersonalAssistantAgent(memory_root=memory_tree.root, llm=llm, text_embedding=text_embedding)
print(agent("Hi! Please tell me my events."))

# Functions for secure data storage and retrieval using RSA key pair
def save_memory_agent(store: LLMsStore, root_node: TextContentNode):
    # Save memory tree embeddings and RSA-encrypted data
    memory_tree = TextMemoryTree(root_node)
    memory_tree.dump_all_embeddings('./tmp/embeddings.json')
    store.set('Memory', root_node.model_dump())
    store.dump_RSA('./tmp/store.rjson', './tmp/public_key.pem')

def load_memory_agent():
    # Load stored RSA-encrypted data and initialize the agent
    store = LLMsStore()
    store.load_RSA('./tmp/store.rjson', './tmp/private_key.pem')
    
    # Retrieve saved model instances
    llm = store.find_all('ChatGPT4oMini:*')[0]
    text_embedding = store.find_all('TextEmbedding3Small:*')[0]
    
    # Reconstruct memory tree from stored data
    root = TextMemoryTree(store.get('Memory'), llm=llm, text_embedding=text_embedding
                         ).load_all_embeddings('./tmp/embeddings.json').root
    agent = PersonalAssistantAgent(memory_root=root, llm=llm, text_embedding=text_embedding)
    return agent, store

# Example usage of saving and loading the memory agent
save_memory_agent(store, root)
agent, store = load_memory_agent()
print(agent("Welcome back! What's planned for today?"))
