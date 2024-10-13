
import json
from LLMAbstractModel.utils import StringTemplate, RegxExtractor
from LLMAbstractModel import LLMsStore,Model4LLMs
descriptions = Model4LLMs.Function.param_descriptions

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

print(workflow.model_dump_json_dict())

# Run the workflow
print(workflow.get_controller().run())

# Retrieve and print the result of each task
print(json.dumps(workflow.model_dump_json_dict(), indent=2))

store.clean()

###############
system_prompt = '''You are an expert in English translation. I will provide you with the text. Please translate it. You should reply with translations only, without any additional information.
## Your Reply Format Example
```translation
...
```'''

vendor = store.add_new_openai_vendor(api_key="OPENAI_API_KEY")
llm = chatgpt4omini = store.add_new_chatgpt4omini(vendor_id='auto', system_prompt=system_prompt)

# Create template and extractor tasks
input_template = store.add_new_function(
    StringTemplate(string='''
```text
{}
```'''))

extract_result = store.add_new_function(RegxExtractor(regx=r"```translation\s*(.*)\s*\n```"))

# Define the workflow with tasks
workflow = store.add_new_workflow(
    tasks={
        input_template.get_id()   : [], 
        llm.get_id()              : [input_template.get_id()],
        extract_result.get_id()   : [llm.get_id()]
    })

# Set input and run the workflow
res = workflow.get_controller().run(input="こんにちは！はじめてのチェーン作りです！")

# Retrieve and print the result
print("Result:", res)
print(json.dumps(workflow.model_dump_json_dict(), indent=2))


# also support seqential list input
workflow = store.add_new_workflow(
    tasks=[
        input_template.get_id(),
        llm.get_id()           ,
        extract_result.get_id(),
    ])

# You can reuse the workflow by setting a new input
res = workflow.get_controller().run(input="常識とは、18歳までに身に付けた偏見のコレクションである。")
print("Result:", res)
print(json.dumps(workflow.model_dump_json_dict(), indent=2))

# save and load workflow
data = store.dumps()
store.clean()
store.loads(data)
workflow = store.find_all('WorkFlow:*')[0]

res = workflow.get_controller().run(input="为政以德，譬如北辰，居其所而众星共之。")
print("Result:", res)
print(json.dumps(workflow.model_dump_json_dict(), indent=2))