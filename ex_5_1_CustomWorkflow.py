
import json
from typing import Optional

from pydantic import BaseModel, Field
from LLMAbstractModel.utils import StringTemplate, RegxExtractor
from LLMAbstractModel import LLMsStore,Model4LLMs
def myprint(string):
    print('##',string,':\n',eval(string),'\n')

store = LLMsStore()

## add custom function
class AddFunction(Model4LLMs.MermaidWorkflowFunction):
    description:str = 'Add two numbers'
    class Arguments(BaseModel):
         x:int=Field(description='first number')
         y:int=Field(description='second number')
    class Returness(BaseModel):
         n:int=Field(description='Added result')
    args:Optional[Arguments]=None
    rets:Optional[Returness]=None
    def __call__(self, x: int, y: int):
        self.args = self.Arguments(x=x,y=y)
        self.rets = self.Returness(n=x+y)
        return self.rets.n

class MultiplyFunction(Model4LLMs.MermaidWorkflowFunction):
    description:str = 'Multiply a number by 2'
    class Arguments(BaseModel):
         x:int=Field(description='number to multiply')
    class Returness(BaseModel):
         n:int=Field(description='doubled number')
    args:Optional[Arguments]=None
    rets:Optional[Returness]=None
    def __call__(self, x: int):
        self.args = self.Arguments(x=x)
        self.rets = self.Returness(n=x * 2)
        return self.rets.n

class SubtractFunction(Model4LLMs.MermaidWorkflowFunction):
    description:str = 'Subtract second number from first'
    class Arguments(BaseModel):
         x:int=Field(description='first number')
         y:int=Field(description='second number')
    class Returness(BaseModel):
         n:int=Field(description='subtracted result')
    args:Optional[Arguments]=None
    rets:Optional[Returness]=None
    def __call__(self, x: int, y: int):
        self.args = self.Arguments(x=x,y=y)
        self.rets = self.Returness(n=x - y)
        return self.rets.n

class ConstantFiveFunction(Model4LLMs.MermaidWorkflowFunction):
    description:str = 'Returness a constant value of 5'
    class Returness(BaseModel):
         n:int=Field(5,description='constant value')
    rets:Returness=Returness()
    def __call__(self):
        return self.rets.n

class ConstantThreeFunction(Model4LLMs.MermaidWorkflowFunction):
    description:str = 'Returness a constant value of 3'
    class Returness(BaseModel):
         n:int=Field(3,description='constant value')
    rets:Returness=Returness()
    def __call__(self):
        return self.rets.n


task_add            = store.add_new_obj(AddFunction())
task_multiply       = store.add_new_obj(MultiplyFunction())
task_constant_three = store.add_new_obj(ConstantThreeFunction())
task_constant_five  = store.add_new_obj(ConstantFiveFunction())


# Create a new WorkFlow instance
workflow:Model4LLMs.MermaidWorkflow = store.add_new_obj(
    Model4LLMs.MermaidWorkflow(
        mermaid_text=f'''
graph TD
    {task_constant_five.get_id()} -- "{{'n':'x'}}" --> {task_multiply.get_id()}
    {task_constant_three.get_id()} -- "{{'n':'x'}}" --> {task_add.get_id()}
    {task_multiply.get_id()} -- "{{'n':'y'}}" --> {task_add.get_id()}
'''))
# parse_mermaid the workflow
myprint('workflow.parse_mermaid()')
## -> 13
# Run the workflow
myprint('workflow.run()')
## -> 13

# # Retrieve and print the result of each task
myprint('json.dumps(workflow.model_dump_json_dict(), indent=2)')
## -> ...

store.clean()

# ###############
system_prompt = '''
You are an expert in English translation.
I will provide you with the text. Please translate it.
You should reply with translations only, without any additional information.
## Your Reply Format Example
```translation
...
```
'''.strip()


vendor = store.add_new_vendor(Model4LLMs.OpenAIVendor)(api_key='OPENAI_API_KEY')
llm = chatgpt = store.add_new_llm(Model4LLMs.ChatGPT41Nano)(vendor_id='auto',system_prompt=system_prompt)

# Create template and extractor tasks
input_template = store.add_new_obj(
    StringTemplate(para=dict(string='''
```text
{text}
```
'''.strip())))

extract_result = store.add_new_obj(RegxExtractor(para=dict(regx=r"```translation\s*(.*)\s*\n```")))

# Define the workflow with tasks
workflow:Model4LLMs.MermaidWorkflow = store.add_new_obj(
    Model4LLMs.MermaidWorkflow(
        mermaid_text=f'''
graph TD
    {input_template.get_id()} -- "{{'data':'messages'}}" --> {llm.get_id()}
    {llm.get_id()} -- "{{'data':'text'}}" --> {extract_result.get_id()}
'''))
workflow.parse_mermaid()
myprint('workflow.run(text="こんにちは！はじめてのチェーン作りです！")["final"]')
# -> {'data': 'Hello! This is my first time making a chain!'}

# save and load workflow
data = store.dumps()
store.clean()
store.loads(data)
workflow = store.find_all('*WorkFlow*')[0]
workflow.parse_mermaid()
myprint('workflow.run(text="为政以德，譬如北辰，居其所而众星共之。")["final"]')
## -> {'data': 'Governing with virtue is like the North Star, remaining in its place while all other stars revolve around it.'}