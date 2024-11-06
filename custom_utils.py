
import datetime
import json
import math
from typing import Optional
from uuid import uuid4
from pydantic import BaseModel, Field
import requests
from LLMAbstractModel.LLMsModel import LLMsStore, Model4LLMs
from LLMAbstractModel.utils import RegxExtractor
descriptions = Model4LLMs.Function.param_descriptions
store = LLMsStore()


##########################
## ex.4.CustomStores.py ##
##########################
class FibonacciObj(Model4LLMs.AbstractObj):
    n:int=1

### registeration magic
store.add_new_obj(FibonacciObj()).get_controller().delete()


############################
## ex.5.CustomWorkflow.py ##
############################
@descriptions('Add two numbers', x='first number', y='second number')
class AddFunction(Model4LLMs.Function):
    def __call__(self, x: int, y: int):
        return x + y

@descriptions('Multiply a number by 2', x='number to multiply')
class MultiplyFunction(Model4LLMs.Function):
    def __call__(self, x: int):
        return x * 2

@descriptions('Subtract second number from first', x='first number', y='second number')
class SubtractFunction(Model4LLMs.Function):
    def __call__(self, x: int, y: int):
        return x - y

@descriptions('Return a constant value of 5')
class ConstantFiveFunction(Model4LLMs.Function):
    def __call__(self):
        return 5

@descriptions('Return a constant value of 3')
class ConstantThreeFunction(Model4LLMs.Function):
    def __call__(self):
        return 3
    
### registeration magic
store.add_new_function(AddFunction()).get_controller().delete()
store.add_new_function(MultiplyFunction()).get_controller().delete()
store.add_new_function(ConstantThreeFunction()).get_controller().delete()
store.add_new_function(ConstantFiveFunction()).get_controller().delete()


#######################
## ex.6.TryAgents.py ##
#######################
@descriptions('Reverse geocode coordinates to an address', lon='longitude', lat='latitude')
class FrenchReverseGeocodeFunction(Model4LLMs.Function):
    def __call__(self, lon: float, lat: float):
        # Construct the URL with the longitude and latitude parameters
        url = f"https://api-adresse.data.gouv.fr/reverse/?lon={lon}&lat={lat}"        
        # Perform the HTTP GET request
        response = requests.get(url)        
        # Check if the request was successful
        if response.status_code == 200:
            # Return the JSON data from the response
            return response.json()
        else:
            # Handle the error case
            return {'error': f"Request failed with status code {response.status_code}"}
        
# Workflow function for querying French address agent and handling responses
@descriptions('Workflow function of querying French address agent', question='The question to ask the LLM')
class FrenchAddressAgent(Model4LLMs.Function):
    french_address_llm_id:str
    first_json_extract:RegxExtractor = RegxExtractor(regx=r"```json\s*(.*)\s*\n```", is_json=True)
    french_address_search_function_id:str
    french_address_system_prompt:str='''
You are familiar with France and speak English.
You will answer questions by providing information.
If you want to use an address searching by coordinates, please only reply with the following text:
```json
{"lon":2.37,"lat":48.357}
```
'''

    def change_llm(self,llm_obj:Model4LLMs.AbstractLLM):
        self.french_address_llm_id = llm_obj.get_id()
        self.get_controller().store()

    def __call__(self,question='I am in France and My GPS shows (47.665176, 3.353434), where am I?',
                 debug=False):
        debugprint = lambda msg:print(f'--> [french_address_agent]: {msg}') if debug else lambda:None
        query = question
        french_address_llm:Model4LLMs.AbstractLLM = self.get_controller().storage().find(self.french_address_llm_id)
        french_address_llm.system_prompt = self.french_address_system_prompt
        french_address_search_function = self.get_controller().storage().find(self.french_address_search_function_id)
        
        while True:
            debugprint(f'Asking french_address_llm with: [{dict(question=query)}]')
            response = french_address_llm(query)
            coord_or_query = self.first_json_extract(response)
            
            # If the response contains coordinates, perform a reverse geocode search
            if isinstance(coord_or_query, dict) and "lon" in coord_or_query and "lat" in coord_or_query:
                debugprint(f'Searching address with coordinates: [{coord_or_query}]')
                query = french_address_search_function(**coord_or_query)
                query = f'\n## Question\n{question}\n## Information\n```\n{query}\n```\n'
            else:
                # Return the final response if no coordinates were found
                answer = coord_or_query
                debugprint(f'Final answer: [{dict(answer=answer)}]')
                return answer

# Workflow for handling triage queries and routing to the appropriate agent
@descriptions('Workflow function of triage queries and routing to the appropriate agent', question='The question to ask the LLM')
class TriageAgent(Model4LLMs.Function):
    triage_llm_id:str
    french_address_agent_id:str
    agent_extract:RegxExtractor = RegxExtractor(regx=r"```agent\s*(.*)\s*\n```")
    triage_system_prompt:str='''
You are a professional guide who can connect the asker to the correct agent.
## Available Agents:
- french_address_agent: Familiar with France and speaks English.

## How to connect the agent:
```agent
french_address_agent
```
'''

    def change_llm(self,llm_obj:Model4LLMs.AbstractLLM):
        self.triage_llm_id = llm_obj.get_id()
        self.get_controller().store()
    
    def __call__(self,
                 question='I am in France and My GPS shows (47.665176, 3.353434), where am I?',
                 debug=False):
        debugprint = lambda msg:print(f'--> [triage_agent]: {msg}') if debug else lambda:None

        triage_llm:Model4LLMs.AbstractLLM = self.get_controller().storage().find(self.triage_llm_id)
        triage_llm.system_prompt = self.triage_system_prompt
        french_address_agent = self.get_controller().storage().find(self.french_address_agent_id)

        debugprint(f'Asking triage_llm with: [{dict(question=question)}]')    
        while True:
            # Get the response from triage_llm and extract the agent name
            response = triage_llm(question)
            agent_name = self.agent_extract(response)
            debugprint(f'agent_extract : [{dict(response=response)}]')
            if 'french_address_agent' in agent_name:
                debugprint(f'Switching to agent: [{agent_name}]')
                return french_address_agent(question,debug=debug)


            
### registeration magic
store.add_new_obj(FrenchReverseGeocodeFunction()).get_controller().delete()
store.add_new_obj(FrenchAddressAgent(french_address_llm_id='',
                                    french_address_search_function_id='')).get_controller().delete()
store.add_new_obj(TriageAgent(triage_llm_id='',french_address_agent_id='')).get_controller().delete()



################################
## ex.7.TryAgentsAndMemory.py ##
################################
# Node model for each node in the memory treefrom typing import list, Optional
class TextContentNode(BaseModel):
    id: str = Field(default_factory=lambda: f"Node:{uuid4()}")
    content: str = "Root(Group)"
    embedding: list[float] = Field(default_factory=list)

    parent_id: str = "NULL"
    children: list['TextContentNode'] = Field(default_factory=list)
    depth: int = 0

    _root_node: 'TextContentNode' = None

    def is_group(self) -> bool:
        return "Group" in self.content

    def is_root(self) -> bool:
        return self.content == "Root(Group)"
    
    def content_with_parents(self):
        groups = [g.content for g in self.parents()]
        content = self.content
        if len(groups)>0:content += f' [{";".join(groups)}]'
        return content

    def content_with_groups(self):
        return self.content_with_parents()
        # groups = [g.content for g in self.groups()]
        # content = self.content
        # if len(groups)>0:content += f' [{";".join(groups)}]'
        # return content

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
    
    def clone(self) -> 'TextContentNode':
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
        p = self.get_parent()
        if p:
            p.children = [c for c in p.children if c.id != self.id]

    def swap(self, other: 'TextContentNode'):
        self.content, other.content = other.content, self.content
        self.embedding, other.embedding = other.embedding, self.embedding

    def add(self, child: 'TextContentNode'):
        self.children.append(child)
        child.parent_id = self.id
        child.depth = self.depth + 1

    def get_parent(self) -> Optional['TextContentNode']:
        return self.get_node(self.parent_id) if self.parent_id else None

    def parents(self) -> list['TextContentNode']:
        path = []
        current = self
        while current and not current.is_root():
            path.append(current)
            current = current.get_parent()
        return list(reversed(path))[:-1]

    def groups(self) -> list['TextContentNode']:
        return [c for c in self.parents() if c.is_group()]

    def get_all_children(self) -> list['TextContentNode']:
        descendants = []
        def collect_children(node: 'TextContentNode'):
            descendants.append(node)
            for child in node.children:
                collect_children(child)
        collect_children(self)
        return descendants

    def move_to(self, new_parent: 'TextContentNode'):
        if self.get_parent():self.dispose()
        new_parent.add(self)
        old_depth = self.depth
        self.depth = new_parent.depth + 1
        for child in self.get_all_children():
            child.depth = child.depth - old_depth + self.depth

    def has_keyword(self, keyword: str) -> bool:
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
        yield node
        for child in node.children:
            yield from self.traverse(child)

    def get_node(self, node_id: str) -> Optional[TextContentNode]:
        return self.root.get_node(node_id)

    def update_content(self, old_content: str, new_content: str):
        if old_content == new_content:return
        for n in self.find_nodes_by_content(old_content):
            n.content = new_content
            n.embedding = []

    def remove_content(self, content: str):
        for node in self.find_nodes_by_content(content):
            node.dispose()

    def find_nodes_by_content(self, keyword: str) -> list[TextContentNode]:
        return [node for node in self.traverse(self.root) if keyword in node.content]

    def get_leaf_nodes(self) -> list[TextContentNode]:
        return [node for node in self.traverse(self.root) if not node.children]

    def reparent_node(self, node_id: str, new_parent_id: str):
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
        res:str = self.llm(f'```text\n{self.print_tree(is_print=False)}\n```')
        res:str = RegxExtractor(regx=r"```text\s*(.*)\s*```")(res)
        root = self.parse_text_tree(res)
        embeddings = {}
        for node in self.traverse(self.root):
            embeddings[node.content_with_groups()] = node.embedding
            node.embedding = []
        
        for node in self.traverse(root):
            node.embedding = embeddings.get(node.content_with_groups(),[])
            
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
                node.embedding = self.text_embedding(node.content_with_groups())
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
    
    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[TextContentNode, float]]:
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
        path = grandchild1.parents()
        assert path[0].content == ["Child 1"]

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

@descriptions('Workflow function of memory agent responsible for generating responses by combining current query input with relevant stored memory. Facilitates memory retrieval, response generation, and memory updating when new information is learned.',
              query='The question to ask the LLM', print_memory='For debugging; displays memory retrieval results')
class PersonalAssistantAgent(Model4LLMs.Function):
    llm_id: str = ''
    text_embedding_id: str = ''
    memory_root: TextContentNode = TextContentNode()
    memory_top_k: int = Field(default=5, ge=1, description="Number of top memories to retrieve.")    
    memory_min_score: float = Field(default=0.4, ge=0.0, description="min similarity of top memories to retrieve.")
    system_prompt: str = "You are a capable, friendly assistant use a mix of past information and new insights to answer effectively. Save new, important details—such as preferences or routines—in your own words. Simply reply with ```memory ... ```."

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
        llm:Model4LLMs.AbstractLLM = self.get_controller().storage().find(self.llm_id)
        llm.system_prompt=self.system_prompt
        response = llm(query)
        if '```memory' in response:
            new_memo = memo_ext(response)
            self.add_memory(new_memo)
        return response

    def get_memory(self):
        llm:Model4LLMs.AbstractLLM = self.get_controller().storage().find(self.llm_id)
        text_embedding:Model4LLMs.AbstractEmbedding = self.get_controller(
                                            ).storage().find(self.text_embedding_id)
        return TextMemoryTree(root=self.memory_root,llm=llm,
                       text_embedding=text_embedding)
        
    def tidy_memory(self):
        self.memory_root = self.get_memory().tidytree().root

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
            res += f"{i}. Score: {score:.3f} | Content: {node.content_with_groups()}\n"
        return res


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
    agent:PersonalAssistantAgent = store.find_all('PersonalAssistantAgent:*')[0]
    agent.load_embeddings('./tmp/embeddings.json')
    return agent, store

### registeration magic
store.add_new_obj(PersonalAssistantAgent()).get_controller().delete()