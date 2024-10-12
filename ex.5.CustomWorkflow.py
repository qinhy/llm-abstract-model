from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Controller4LLMs, Model4LLMs
descriptions = Model4LLMs.Function.param_descriptions

store = LLMsStore()

system_prompt = 'You are smart assistant'

vendor = store.add_new_openai_vendor(api_key='OPENAI_API_KEY')
chatgpt4omini = store.add_new_chatgpt4omini(vendor_id='auto',system_prompt=system_prompt)

# vendor  = store.add_new_ollama_vendor()
# gemma2  = store.add_new_gemma2(vendor_id=vendor.get_id(),system_prompt=system_prompt)
# phi3    = store.add_new_phi3(vendor_id=vendor.get_id(),system_prompt=system_prompt)
# llama32 = store.add_new_llama(vendor_id=vendor.get_id(),system_prompt=system_prompt)

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
workflow:Model4LLMs.WorkFlow = store.add_new_obj(Model4LLMs.WorkFlow(tasks = {
    task_add.get_id()           : [task_multiply.get_id(), task_constant_three.get_id()],
    task_multiply.get_id()      : [task_constant_five.get_id()],             
    task_constant_three.get_id(): [],                         
    task_constant_five.get_id() : []                          
}))

print(workflow.model_dump_json_dict())

# Run the workflow
print(workflow.get_controller().run())

# Retrieve and print the result of each task
print("Results :", workflow.results)

print(store.dumps())