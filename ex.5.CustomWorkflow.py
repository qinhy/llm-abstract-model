
import json
from LLMAbstractModel.utils import StringTemplate, RegxExtractor
from LLMAbstractModel import LLMsStore,Model4LLMs
descriptions = Model4LLMs.Function.param_descriptions
def myprint(string):
    print('##',string,':\n',eval(string),'\n')

store = LLMsStore()

## add custom function
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


task_add            = store.add_new_function(AddFunction())
task_multiply       = store.add_new_function(MultiplyFunction())
task_constant_three = store.add_new_function(ConstantThreeFunction())
task_constant_five  = store.add_new_function(ConstantFiveFunction())

# Create a new WorkFlow instance
workflow:Model4LLMs.WorkFlow = store.add_new_obj(
            Model4LLMs.WorkFlow(tasks = {
                    task_add.get_id()           : [task_multiply.get_id(), task_constant_three.get_id()],
                    task_multiply.get_id()      : [task_constant_five.get_id()],             
                    task_constant_three.get_id(): [],                         
                    task_constant_five.get_id() : []                          
        }))

# Run the workflow
myprint('workflow.get_controller().run()')
## -> 13

# Retrieve and print the result of each task
myprint('json.dumps(workflow.model_dump_json_dict(), indent=2)')
## -> ...

store.clean()

###############
system_prompt = '''
You are an expert in English translation.
I will provide you with the text. Please translate it.
You should reply with translations only, without any additional information.
## Your Reply Format Example
```translation
...
```
'''

vendor = store.add_new_openai_vendor(api_key="OPENAI_API_KEY")
llm = chatgpt4omini = store.add_new_chatgpt4omini(vendor_id='auto', system_prompt=system_prompt)

# Create template and extractor tasks
input_template = store.add_new_function(
    StringTemplate(string='''
```text
{text}
```'''))

extract_result = store.add_new_function(RegxExtractor(regx=r"```translation\s*(.*)\s*\n```"))

# Define the workflow with tasks
workflow = store.add_new_workflow(
    tasks={
        input_template.get_id()   : ['input'],
        llm.get_id()              : [input_template.get_id()],
        extract_result.get_id()   : [llm.get_id()]
    })
# input=[args,kwargs]
myprint('workflow(input=[(),dict(text="こんにちは！はじめてのチェーン作りです！")])')
## -> Hello! This is my first time making a chain!

workflow.get_controller().delete()
input_template.get_controller().delete()

# also support seqential list input
input_template = store.add_new_function(
    StringTemplate(string='''
```text
{}
```'''))
workflow = store.add_new_workflow(
    tasks=[
        input_template.get_id(),#   : [],
        llm.get_id(),#              : [input_template.get_id()],
        extract_result.get_id(),#   : [llm.get_id()]
    ])
# You can reuse the workflow by setting a new input
myprint('workflow("常識とは、18歳までに身に付けた偏見のコレクションである。")')
## -> Common sense is a collection of prejudices acquired by the age of 18.

# save and load workflow
data = store.dumps()
store.clean()
store.loads(data)
workflow = store.find_all('WorkFlow:*')[0]
myprint('workflow("为政以德，譬如北辰，居其所而众星共之。")')
## -> Governing with virtue is like the North Star, which occupies its position while all the other stars revolve around it.