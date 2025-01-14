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
    
class HistoryAssistantAgent(BaseModel):
    llm: Model4LLMs.AbstractLLM = None
    # text_embedding: Model4LLMs.AbstractEmbedding = None
    history_root: TextContentNode = TextContentNode()
    history_last: TextContentNode = None
    # history_last_k: int = Field(default=5, ge=1, description="Number of last history to retrieve.")

    def print_tree(self, node: Optional[TextContentNode] = None, level: int = 0, is_print=True):
        res = ''
        if node is None: node = self.history_root
        # Print current node with indentation corresponding to its depth
        res += " " * (level * 4) + f"- {node.content.strip()}\n"
        for child in node.children:
            res += self.print_tree(child, level + 1)
        if node == self.history_root:
            if is_print:print(res[:-1])
            res = res[:-1]
        return res

    def get_last_history(self):
        if self.history_last is None:
            self.history_last = self.history_root
        return self.history_last

    def add_history(self, content: str, role:str='user'):
        p = self.get_last_history().get_parent()
        if p is None:p = self.get_last_history()
        last = TextContentNode(content=role+'@@@@'+content)
        p.add(last)
        self.history_last = last
    
    def history_retrieval(self, last_k: int=4):
        p = self.get_last_history()
        return p.children[-last_k:]

    def __call__(self, qustion: str, system_prompt:str=None, last_k: int=4, print_history=True) -> str:
        
        if print_history:
            print("############ For Debug ##############")
            self.print_history()
            print("#####################################")
            print()

        history = self.history_retrieval(last_k)
        tmp = self.llm.system_prompt
        self.llm.system_prompt = system_prompt
        response = self.llm([
                {'role':h.content.split('@@@@')[0],
                 'content':'@@@@'.join(h.content.split('@@@@')[1:])} for h in history
            ]+{'role':'user','content':qustion})
        self.add_history(qustion)
        self.add_history(response,'assistant')
        self.llm.system_prompt = tmp
        return response

# Functions for secure data storage and retrieval using RSA key pair
def save_history_agent(store: LLMsStore, root_node: TextContentNode):
    store.set('history', root_node.model_dump())
    store.dump_RSA('./tmp/store.rjson', './tmp/public_key.pem',True)

def load_history_agent():
    # Load stored RSA-encrypted data and initialize the agent
    store = LLMsStore()
    store.load_RSA('./tmp/store.rjson', './tmp/private_key.pem')
    
    # Retrieve saved model instances
    llm = store.find_all('ChatGPT4oMini:*')[0]
    
    # Reconstruct history tree from stored data
    agent = HistoryAssistantAgent(history_root=store.get('history'), llm=llm)
    return agent, store

# Example usage of saving and loading the memory agent
# save_memory_agent(store, agent.memory_root)
# agent, store = load_memory_agent()
# print(agent("Welcome back! What's planned for today?"))
