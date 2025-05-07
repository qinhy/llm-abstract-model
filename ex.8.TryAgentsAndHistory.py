import json
from typing import List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field
from LLMAbstractModel.LLMsModel import LLMsStore, Model4LLMs

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
        if self.is_root():
            child._root_node = self
        elif self._root_node:
            child._root_node = self._root_node

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

    def history_retrieval(self, last_k: int=4):
        p = self.get_last_history()
        res = p.parents() + [p]
        return [i for i in res if not i.is_root()][-last_k:]

    def add_history(self, content: str, role:str='user',name:str=''):
        if content is None:content='NULL'
        content=role+'@@@@'+content
        if name:
            content = content+'@@@@'+name
        last = TextContentNode(content=content)
        self.get_last_history().add(last)
        self.history_last = last
        # {role}@@@@{content}
        # {role}@@@@{content}@@@@{name}

    def add_funcalls_history(self,tool_calls: str,
                            content: str='NULL', role:str='assistant'):
        self.add_history(content,role)
        last = self.get_last_history()
        last.content += '@@tc@@'+tool_calls
        # assistant@@@@{content}@@tc@@{tool_calls}
        # assistant@@@@{content}@@@@{name}@@tc@@{tool_calls}
        
    def add_funres_history(self,name: str,tool_call_id: str,
                            content: str='NULL', role:str='tool'):
        self.add_history(content,role,name)
        last = self.get_last_history()
        last.content += '@@tcid@@'+tool_call_id
        # tool@@@@{content}@@@@{name}@@tcid@@{tool_call_id}

    def prepare_openai_his_messages(self, history: List[TextContentNode]):
        msgs = []
        for h in history:
            if '@@@@' in h.content:
                cs = h.content.split('@@@@')
                msgs.append({'role':cs[0],'content':cs[1]})
                if len(cs)>2:
                    msgs[-1]['name'] = cs[2]
                
            if '@@tcid@@' in h.content:
                i = h.content.split('@@tcid@@')[1]
                msgs[-1]['tool_call_id'] = i

            if '@@tc@@' in h.content:
                c = h.content.split('@@tc@@')[1]
                msgs[-1]['tool_calls'] = json.loads(c)
                
            if 'NULL@@' in msgs[-1]['content']:
                msgs[-1]['content'] = ''
                
        return msgs

    def __call__(self, qustion: str, system_prompt:str=None, last_k: int=4, print_history=True) -> str:
        tmp = self.llm.system_prompt
        self.llm.system_prompt = system_prompt

        history = self.history_retrieval(last_k)
        msgs = self.prepare_openai_his_messages(history)
        msgs.append({'role':'user','content':qustion})

        if print_history:
            print("############ For Debug ##############")
            self.print_tree()
            print("#####################################")
            [print(i) for i in msgs]
            print("#####################################")

        self.add_history(qustion)
        response = self.llm(msgs)
        if 'calls' in response:
            self.add_funcalls_history(
                json.dumps(response['calls']),response['content'])
        else:
            self.add_history(response,'assistant')        

        self.llm.system_prompt = tmp
        return response

# Functions for secure data storage and retrieval using RSA key pair
def save_history_agent(store: LLMsStore, root_node: TextContentNode):
    store.set('history', root_node.model_dump())
    store.dump_RSA('./tmp/TryAgentsAndHistory.rjson', './tmp/public_key.pem',True)

def load_history_agent():
    # Load stored RSA-encrypted data and initialize the agent
    store = LLMsStore()
    store.load_RSA('./tmp/TryAgentsAndHistory.rjson', './tmp/private_key.pem')
    
    # Retrieve saved model instances
    llm = store.find_all('ChatGPT*:*')[0]
    
    # Reconstruct history tree from stored data
    agent = HistoryAssistantAgent(history_root=store.get('history'), llm=llm)
    return agent, store

# Example usage of saving and loading the history agent
# save_history_agent(store, agent.history_root)
# agent, store = load_history_agent()
# print(agent("Welcome back! What's planned for today?"))

store = LLMsStore()
vendor = store.add_new_vendor(
    Model4LLMs.OpenAIVendor)(api_key='OPENAI_API_KEY')
llm = store.add_new_llm(
    Model4LLMs.ChatGPT41Nano)(vendor_id=vendor.get_id(), temperature=0.7)
agent = HistoryAssistantAgent(llm=llm)