import datetime
import json
import math
from typing import List, Optional
from uuid import uuid4
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

    def is_group(self) -> bool:
        return self.content[-7:] == "(Group)"

    def is_root(self) -> bool:
        return self.content == "Root(Group)"
    
    def content_with_parents(self):
        groups = [g.content for g in self.parents()]
        content = self.content
        if len(groups)>0:content += f' [{";".join(groups)}]'
        return content

    def reflesh_root(self):
        if self.is_root():
            for child in self.traverse(self):
                if not child.is_root():
                    child._root_node = self

    def traverse(self, node: 'TextContentNode'):
        yield node
        for child in node.children:
            yield from self.traverse(child)

    def get_node(self, node_id: str) -> Optional['TextContentNode']:
        root = self if self.is_root() else self._root_node
        if root is None: return None
        for child in root.traverse(root):
            if child.id == node_id:
                return child
        # # If no node found, return None
        return None
    
    def dispose(self):
        p = self.get_parent()
        if p:
            p.children = [c for c in p.children if c.id != self.id]

    def add(self, child: 'TextContentNode'):
        self.children.append(child)
        child.parent_id = self.id
        child.depth = self.depth + 1

    def get_parent(self) -> Optional['TextContentNode']:
        return self.get_node(self.parent_id) if self.parent_id else None

    def parents(self) -> List['TextContentNode']:
        path = []
        current = self
        while current and not current.is_root():
            path.append(current)
            current = current.get_parent()
        return list(reversed(path))[:-1]

    def groups(self) -> List['TextContentNode']:
        return [c for c in self.parents() if c.is_group()]

    def get_all_children(self) -> List['TextContentNode']:
        descendants = []
        def collect_children(node: 'TextContentNode'):
            descendants.append(node)
            for child in node.children:
                collect_children(child)
        collect_children(self)
        return descendants
    
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
        yield node
        for child in node.children:
            yield from self.traverse(child)

    def remove_content(self, content: str):
        for node in self.find_nodes_by_content(content):
            node.dispose()

    def find_nodes_by_content(self, keyword: str) -> List[TextContentNode]:
        return [node for node in self.traverse(self.root) if keyword in node.content]

    def classification(self, src: TextContentNode, ref: TextContentNode):
        sys = "## Instructions:¥nGiven two pieces of content, 'ref' and 'src', identify their relationship and label it with one of the following:¥nProvide only a single word as your output: Child, Parent, or Isolate.¥n¥n## Procedure:¥n1. Classify Each Content:¥n   - Determine whether 'ref' and 'src' are a topic or an action.¥n     - *Topic*: A general subject area.¥n     - *Action*: A specific activity within a topic.¥n¥n2. Assess Connection:¥n   - Evaluate whether there is any connection between 'ref' and 'src', considering both direct and indirect links.¥n¥n3. Apply Inclusion Rule:¥n   - Remember that a topic always includes its related actions.¥n¥n4. Compare Topics (if both are topics):¥n   - Determine if one topic includes the other.¥n¥n5. Select the Appropriate Label:¥n   - Choose Child if 'src' potentially includes 'ref'.¥n   - Choose Parent if 'ref' potentially includes 'src'.¥n   - Choose Isolate if no connection exists.¥n¥n## Notes:¥n- Carefully determine whether each piece of content is a topic or an action.¥n- Consider subtle connections for accurate labeling.¥n"
        self.llm.system_prompt=sys
        res:str = self.llm(f'## src¥n{src.content}¥n## ref¥n{ref.content}')
        return {'parent':-1,'isolate':0,'child':1}[res.strip().lower()]

    def tidytree(self,res=None):
        if res is None:
            sys = '''Please organize the following memory list and respond in the original tree structure format. If a node represents a group, add '(Group)' at the end of the node name. Feel free to edit, delete, move or add new nodes (or groups) as needed. Start with the root node and add child nodes at the appropriate level.'''
            self.llm.system_prompt=sys
            res:str = self.llm(f'```text\n{self.print_tree(is_print=False)}\n```')
            res:str = RegxExtractor(regx=r"```text\s*(.*)\s*```")(res)
            # print(res)
        root = self.parse_text_tree(res)
        embeddings = {}
        for node in self.traverse(self.root):
            embeddings[node.content_with_parents()] = node.embedding
            node.embedding = []
        
        for node in self.traverse(root):
            node.embedding = embeddings.get(node.content_with_parents(),[])
            
        self.root = root
        return self
        
    def get_threshold(self, depth: int) -> float:
        return self.base_threshold * math.exp(self.lambda_factor * depth)
    
    def calc_embedding(self, node:TextContentNode):
        if len(node.embedding)>0:return node
        if node.is_group():
            node.embedding = [0.0] * self.text_embedding.embedding_dim
        else:
            if node.content in TextMemoryTree._embedding_cache_dict:
                node.embedding = TextMemoryTree._embedding_cache_dict[node.content]
            if len(node.embedding)==0:
                node.embedding = self.text_embedding(node.content_with_parents())
        return node

    def similarity(self, src: TextContentNode, ref: TextContentNode)->float:
        # Cosine similarity calculation
        emb1, emb2 = self.calc_embedding(ref).embedding, self.calc_embedding(src).embedding
        norm_emb1 = math.sqrt(sum(x ** 2 for x in emb1))
        norm_emb2 = math.sqrt(sum(x ** 2 for x in emb2))

        norm = norm_emb1 * norm_emb2
        if norm == 0: return 0.0
        dot_product = sum(x * y for x, y in zip(emb1, emb2))

        coss = dot_product / norm
        return coss
    
    def extract_all_embeddings(self):
        embeddings:dict[str,list[float]] = {}
        for node in self.traverse(self.root):
            embeddings[node.id] = node.embedding
            node.embedding = []
        return embeddings
    
    def dump_all_embeddings(self,path='embeddings.json'):
        embeddings = self.extract_all_embeddings()
        with open(path, "w") as tf: tf.write(json.dumps(embeddings))
    
    def put_all_embeddings(self,embeddings:dict[str,list[float]]):
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
        new_node = self.calc_embedding(TextContentNode(content=content.strip()))
        self._insert_node(self.root, new_node)
        print(f'[TextMemoryTree]: insert new content [{content.strip()}]')
        # self.print_tree()
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
        text = text.replace('- ','')
        lines = text.splitlines()
        # Skip empty lines
        lines = [line for line in lines if line.lstrip()]
        
        stack = [(TextContentNode(content=lines.pop(0)), -1)]
        root = stack[0][0]


        if not root.is_root():raise ValueError('first Node must be Root.')        

        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            content = stripped.strip()  # Remove bullet and any additional whitespace
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
    
    def print_tree(self, node: Optional[TextContentNode] = None, level: int = 0, is_print=True):
        res = ''
        if node is None: node = self.root
        # Print current node with indentation corresponding to its depth
        res += " " * (level * 4) + f"- {node.content.strip()}\n"
        for child in node.children:
            res += self.print_tree(child, level + 1)
        if node == self.root:
            if is_print:print(res[:-1])
            res = res[:-1]
        return res
    
class PersonalAssistantAgent(BaseModel):
    llm: Model4LLMs.AbstractLLM = None
    text_embedding: Model4LLMs.AbstractEmbedding = None
    memory_root: TextContentNode = TextContentNode()
    memory_top_k: int = Field(default=5, ge=1, description="Number of top memories to retrieve.")    
    memory_min_score: float = Field(default=0.3, ge=0.0, description="min similarity of top memories to retrieve.")
    system_prompt: str = '''You are a capable and friendly assistant, combining past knowledge with new insights to provide effective answers.  
Note: When it is necessary to retain new and important information, such as preferences or routines, you may respond using a block of ```memory ...```."
'''

    def get_memory(self):
        return TextMemoryTree(root=self.memory_root,llm=self.llm,
                       text_embedding=self.text_embedding)
        
    def tidy_memory(self,res=None):
        self.memory_root = self.get_memory().tidytree(res).root

    def print_memory(self):
        self.get_memory().print_tree()

    def add_memory(self, content: str):
        self.get_memory().insert(content)
    
    def load_embeddings(self,path='embeddings.json'):
        self.get_memory().load_all_embeddings(path)
    
    def memory_retrieval(self, query: str) -> str:
        # Perform retrieval and format the results
        res = f"## Memories for the query:\n"
        results = self.get_memory().retrieve(query, self.memory_top_k)
        if len(results)==0:return res+"No memories.\n"
        for i, (node, score) in enumerate(results, start=1):
            if score < self.memory_min_score:continue
            res += f"{i}. Score: {score:.3f} | Content: {node.content_with_parents()}\n"
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

# Functions for secure data storage and retrieval using RSA key pair
def save_memory_agent(store: LLMsStore, root_node: TextContentNode):
    # Save memory tree embeddings and RSA-encrypted data
    memory_tree = TextMemoryTree(root_node)
    memory_tree.dump_all_embeddings('./tmp/embeddings.json')
    store.set('Memory', root_node.model_dump())
    store.dump_RSA('./tmp/store.rjson', './tmp/public_key.pem',True)

def load_memory_agent():
    # Load stored RSA-encrypted data and initialize the agent
    store = LLMsStore()
    store.load_RSA('./tmp/store.rjson', './tmp/private_key.pem')
    
    # Retrieve saved model instances
    llm = store.find_all('ChatGPT4oMini:*')[0]
    text_embedding = store.find_all('TextEmbedding3Small:*')[0]
    
    # Reconstruct memory tree from stored data
    agent = PersonalAssistantAgent(memory_root=store.get('Memory'), llm=llm,
                                   text_embedding=text_embedding, memory_top_k=10)
    agent.load_embeddings('./tmp/embeddings.json')
    return agent, store

# Example usage of saving and loading the memory agent
# save_memory_agent(store, agent.memory_root)
# agent, store = load_memory_agent()
# print(agent("Welcome back! What's planned for today?"))


# main usage form here
# Sample queries reflecting various personal scenarios
queries = [
    "Basic Info: Name - Alex Johnson, Birthday - 1995-08-15, Phone - +1-555-1234, Email - alex.johnson@email.com, Address - 123 Maple Street, Springfield",
    "Personal Details: Occupation - Software Developer, Hobbies - reading, hiking, coding, photography",
    "Friends: Taylor Smith (Birthday: 1994-02-20, Phone: +1-555-5678), Jordan Lee (Birthday: 1993-11-30, Phone: +1-555-9101), Morgan Brown (Birthday: 1996-05-25, Phone: +1-555-1213)",
    "Work & Goals: Company - Tech Solutions Inc., Position - Front-End Developer, Work Email - alex.j@techsolutions.com, Work Phone - +1-555-4321, Goals - Learn a new programming language, Complete a marathon, Read 20 books this year"
]

# Initialize the LLM Store and vendor
store = LLMsStore()
vendor = store.add_new_vendor(Model4LLMs.OpenAIVendor)(api_key='OPENAI_API_KEY')

# Add the necessary components
text_embedding = store.add_new_obj(Model4LLMs.TextEmbedding3Small())
llm = store.add_new_llm(Model4LLMs.ChatGPT41Nano)(vendor_id='auto',temperature=0.7)

# vendor = store.add_new_Xai_vendor(api_key='XAI_API_KEY')
# llm = grok = store.add_new_grok(vendor_id=vendor.get_id())

# Create the root node and initialize the memory tree
root = TextContentNode()
memory_tree = TextMemoryTree(root, llm=llm, text_embedding=text_embedding)

# Insert complex sample data into the memory tree
for memory in queries: memory_tree.insert(memory)

# Print the tree structure to visualize the organization
print("\n########## Memory Tree Structure:")
memory_tree.print_tree()

print("\n########## Tidied Memory Tree Structure:")
memory_tree.tidytree().print_tree()

# Define a function to test memory retrieval with various sample queries
def test_retrieval(memory_tree: TextMemoryTree, query: str, top_k: int = 5):
    print(f"\nTop {top_k} matches for query: '{query}'")
    results = memory_tree.retrieve(query, top_k)
    for i, (node, score) in enumerate(results, start=1):
        print(f"{i}. score: {score:.4f} | [ {node.content} ]")

# Run retrieval tests
questions = [
    # Basic Info
    "What is Alex Johnson's full name and birthday?",
    # Personal Details
    "What is Alex's occupation?",
    # Friends
    "Who are Alex's friends, and when are their birthdays?",
    # Work & Goals
    "Where does Alex work and what is their position?",
]
for query in questions:
    test_retrieval(memory_tree, query, top_k=6)

# Initialize the personal assistant agent using the memory tree
agent = PersonalAssistantAgent(memory_root=memory_tree.root,
                               llm=llm, text_embedding=text_embedding)
print(agent("Hi! Please tell me Taylor info."))
