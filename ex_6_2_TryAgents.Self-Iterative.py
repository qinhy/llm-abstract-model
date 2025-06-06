
from LLMAbstractModel.utils import RegxExtractor
from LLMAbstractModel import LLMsStore,Model4LLMs
descriptions = Model4LLMs.Function.param_descriptions
def myprint(string):
    print('##',string,':\n',eval(string),'\n')

store = LLMsStore()
vendor = store.add_new_vendor(Model4LLMs.OpenAIVendor)(api_key='OPENAI_API_KEY',timeout=60)
debug = True

# enhance from https://github.com/richards199999/Self-Iterative-Agent-System-for-Complex-Problem-Solving/tree/main

## add Main model
@descriptions('Workflow function of main model of Self-Iterative-Agent-System-for-Complex-Problem-Solving',
               question='The question to ask the LLM')
class MainAgent(Model4LLMs.Function):
    solutions:int = 2
    each_iterations:int = 2
    llm_id:str
    eval_agent_id:str
    relevant_concepts_extract:RegxExtractor = RegxExtractor(regx=r"<relevant_concepts>\s*(.*)\s*</relevant_concepts>")
    thoughts_extract:RegxExtractor = RegxExtractor(regx=r"<thoughts>\s*(.*)\s*</thoughts>")
    process_extract:RegxExtractor = RegxExtractor(regx=r"<process>\s*(.*)\s*</process>")
    solution_extract:RegxExtractor = RegxExtractor(regx=r"<solution>\s*(.*)\s*</solution>")
    system_prompt: str = '''**You will be addressing questions and tasks as a professional problem-solver. When the user provides a task or question, follow these steps:**

### Step 1: Identify Relevant Concepts
Carefully read the question or task to identify the key concepts, principles, and knowledge areas required to address it. Write these concepts inside `<relevant_concepts>` tags.  

### Step 2: Brainstorm and Plan
Brainstorm ideas and approaches to solve the task, document every step in the following specified format. Explore multiple perspectives and think creatively, considering out-of-the-box solutions. Flexibly apply relevant principles, tools, or methodologies to guide your thinking. Write your detailed thoughts, insights, and potential strategies within <thoughts> tags.
<thoughts>
...
## Step n
### Observation:
*Your Observed information from user message or previous steps*
### Thought:
*Your consideration based on the above observation*
### Action:
*Your recommended action based on the above thought*
...
</thoughts>

### Step 3: Execute the Solution  
Start executing the solution based on your brainstorming. As you work through the task, document all steps in detail inside `<process>` tags. Provide **a comprehensive breakdown of all actions, reasoning, and steps involved**, ensuring clarity and accuracy. Break down complex steps into simpler components, and DO NOT skip or omit any part, even if it seems obvious or easy. Maintain clear formatting and structure for the process.  

### Step 4: Present the Final solution  
After completing the task, present your final outcome inside `<solution>` tags. Ensure the solution is **clearly formatted, concise, and includes any necessary units, labels, or context**.  

### Additional Notes
Remember to **maintain a professional, thorough, and precise approach** throughout. Your goal is to **provide a well-explained, accurate, and actionable solution** for any task or question presented.

*Note:* Your entire process will be sent to another assistant for evaluation, so refine your approach based on any detailed feedback you receive.
'''
    final_prompt: str = r'''You will be evaluating a set of solutions to a problem and determining the best solution among them. Your task is to evaluate each solution based on various criteria and select the best one. Follow these steps:

- Read through each solution carefully, paying attention to the relevant concepts, thoughts, processes, and final solution. Assess the clarity, logic, and organization of each solution.

- Assess the reasoning and logic behind each solution. Determine how well the thoughts are explained. Assign a score from 1 to 5 for reasoning and logic, where 5 represents the most coherent and logical approach.

- Evaluate the processes in each solution. Check for any errors, inconsistencies, or step incompleteness in the problem-solving process. Assign a score from 1 to 5 for these factors, where 5 represents the highest accuracy.

- Consider the clarity and presentation of each solution. Evaluate how well the solution is structured, how easy it is to follow, and whether the final solution is presented in the correct format. Assign a score from 1 to 5 for clarity and presentation, where 5 represents the most clear and well-presented solution.

- Based on the scores assigned for process accuracy, reasoning and logic, and clarity and presentation, determine an overall score for each solution. The overall score should be a weighted average, with process completeness being the most important factor, followed by reasoning and logic, and then clarity and presentation.

Present your selection using the following template:
```
## Solution 0
process: {score:.2f}, reasoning_and_logic: {score:.2f}, clarity_and_presentation: {score:.2f}, overall_score: {score:.2f}
## Solution 1
process: {score:.2f}, reasoning_and_logic: {score:.2f}, clarity_and_presentation: {score:.2f}, overall_score: {score:.2f}

The best solution is: {solution_number}
Justification: (Provide a brief explanation of why this solution was selected as the best)
Final solution: (Present final solution inside `<solution>` tags)
```
Remember to be objective, thorough, and consistent in your evaluation. Your goal is to identify the solution that demonstrates the highest level of process, logical reasoning, and clear presentation.
'''

    def change_llm(self, llm_obj: Model4LLMs.AbstractLLM):
        self.ll_id = llm_obj.get_id()
        self.controller.store()
        
    def __call__(self, question, debug=False):
        debug_print = lambda msg: print(f'--> [main_agent]: {msg}') if debug else lambda: None

        main_llm: Model4LLMs.AbstractLLM = self.controller.storage().find(self.llm_id)
        eval_agent = self.controller.storage().find(self.eval_agent_id)

        main_llm = main_llm.model_copy()
        main_llm.system_prompt = self.system_prompt

        final_llm = main_llm.model_copy()
        final_llm.system_prompt = self.final_prompt

        debug_print(f'Asking main_llm with: [{dict(question=question)}]')
        def get_review(answer):
            return eval_agent(
                question,
                self.relevant_concepts_extract(answer),
                self.thoughts_extract(answer),
                self.process_extract(answer),
                self.solution_extract(answer),
                debug=debug
            )
        
        def thinking(question, solution=None, review=None):
            question_tmp = ('Please answer the following question based on the previous solution and review.\n' if solution else '') + (
                           '## question\n{}\n') + (
                           '## solution\n{}\n' if solution else '') + (
                           '## review\n{}\n' if review else '')
            ask = question_tmp.format(question, solution, review)
            answer = main_llm(ask)
            print(f'############# thoughts ##############')
            print(self.thoughts_extract(answer))
            print(f'############# solution ##############')
            print(self.solution_extract(answer))

            solution = self.solution_extract(answer)
            process = self.process_extract(answer)
            review = get_review(answer)
            return question, ask, answer, process, solution, review

        rs = []
        for ii in range(self.solutions):
            solution = None
            review = None
            for i in range(self.each_iterations):
                print(f'############# ite=={i} ##############')
                print((question, solution, review))
                question, ask, answer, process, solution, review = thinking(question, solution, review)
                print(review)
                debug_print(ask)
                
            rs.append(f'### Solution {ii}\n' + f'\n```solution\n{solution}\n```' + f'\n```explain\n{process}\n```')
            print(f'############# solution=={ii} ##############')
            print("\n" + rs[-1])

        rs = "\n".join(rs)        
        print(f'############# final ##############')
        print(f'## question\n{question}\n## solutions\n{rs}\n')
        answer = final_llm(f'## question\n{question}\n## solutions\n{rs}\n')
        return answer,self.solution_extract(answer)


@descriptions('Workflow function of eval model of Self-Iterative-Agent-System-for-Complex-Problem-Solving',
              question='The question to ask the LLM')
class EvalAgent(Model4LLMs.Function):
    llm_id: str
    initial_review_extract: RegxExtractor = RegxExtractor(regx=r"<initial_review>\s*(.*)\s*</initial_review>")
    reasoning_feedback_extract: RegxExtractor = RegxExtractor(regx=r"<reasoning_feedback>\s*(.*)\s*</reasoning_feedback>")
    process_errors_extract: RegxExtractor = RegxExtractor(regx=r"<process_errors>\s*(.*)\s*</process_errors>")
    overall_assessment_extract: RegxExtractor = RegxExtractor(regx=r"<overall_assessment>\s*(.*)\s*</overall_assessment>")
    system_prompt: str = '''You will review the problem-solving process of another assistant that has answered a question. Your task is to evaluate the solution and provide a detailed review for refinement. Follow these steps:

### step1
Carefully read through the original question and the entire solution, paying close attention to the relevant concepts, reasoning, processes, and the final solution. Assess whether the solution is clear, logical, and well-organized. Write your initial review within `<initial_review>` tags.  

### step2
Evaluate the reasoning and logic behind the solution. Ensure that the reasoning is clear and coherent. If you find any areas that need clarification or improvement, provide your suggestions within `<reasoning_feedback>` tags.  

### step3
Re-do the processes presented in the `<process>` section **carefully and step-by-step** to verify their accuracy. Break the processes down into the simplest possible steps and check each step for errors. You must treat each part with rigor and avoid carelessness. Ensure that no part of the solution process is neglected during verification. If you find any mistakes, document them within `<process_errors>` tags.  

### step4
Provide an overall assessment of the solution's thoroughness, accuracy, and clarity within `<overall_assessment>` tags. Highlight the strengths and weaknesses of the solution and offer suggestions for improvement, if any.  

Remember to be thorough, constructive, and professional in your review. Your goal is to help improve the quality and accuracy of the problem-solving process.'''

    def change_llm(self,llm_obj:Model4LLMs.AbstractLLM):
        self.ll_id = llm_obj.get_id()
        self.controller.store()
    
    def __call__(self,question,relevant_concepts,thoughts,process,solution,
                 debug=True):
        debugprint = lambda msg:print(f'--> [eval_agent]: {msg}') if debug else lambda:None
        question = ('## question\n{}\n' + (
                    '## relevant concepts\n{}\n' if relevant_concepts else '') + (
                    '## thoughts\n{}\n' if thoughts else '') + (
                    '## process\n{}\n' if process else '') + (
                    '## solution\n{}\n' if solution else '')).format(
                        question,relevant_concepts,thoughts,process,solution)
                
        eval_llm:Model4LLMs.AbstractLLM = self.controller.storage().find(self.llm_id).model_copy()
        eval_llm.system_prompt = self.system_prompt
        debugprint(f'Asking eval_llm with: [{dict(question=question)}]')
        answer = eval_llm(question)
        # debugprint(f'initial_review == [{self.initial_review_extract(answer)}]')
        # debugprint(f'reasoning_feedback == [{self.reasoning_feedback_extract(answer)}]')
        # debugprint(f'process_errors == [{self.process_errors_extract(answer)}]')
        # debugprint(f'overall_assessment == [{self.overall_assessment_extract(answer)}]')
        return f'### initial review\n{self.initial_review_extract(answer)}\n### reasoning feedback\n{self.reasoning_feedback_extract(answer)}\n### process errors\n{self.process_errors_extract(answer)}\n### overall assessmt\n{self.overall_assessment_extract(answer)}'

main_llm = store.add_new_llm(Model4LLMs.ChatGPT41Nano)(vendor_id='auto',limit_output_tokens=4096)
eval_llm = store.add_new_llm(Model4LLMs.ChatGPT41Nano)(vendor_id='auto',limit_output_tokens=4096)

# main_llm = store.add_new_grok(vendor_id='auto',limit_output_tokens=4096)
# eval_llm = store.add_new_grok(vendor_id='auto',limit_output_tokens=4096)

# main_llm = store.add_new_deepseek(vendor_id='auto',limit_output_tokens=4096)
# eval_llm = store.add_new_deepseek(vendor_id='auto',limit_output_tokens=4096)

store.add_new_obj(EvalAgent(llm_id=eval_llm.get_id()),id='EvalAgent:eval_agent')
store.add_new_obj(MainAgent(llm_id=main_llm.get_id(),eval_agent_id='EvalAgent:eval_agent'),id='MainAgent:main_agent')

data = store.dumps()
store.clean()
store.loads(data)

# Example usage
question='How many characters of the letter "r" in "raspberrrry"?'
print(f'\n\n######## Question ########\n\n{question}')
answer,solution = store.find('MainAgent:main_agent')(question=question,debug=False)
print(f'\n\n######## Answer ########\n\n{answer}')
print(f'\n\n######## Solution ########\n\n{solution}')

# Example usage
# question='''Please organize the following memory list and respond in the original tree structure format. If a node represents a group, add '(Group)' at the end of the node name. Feel free to add new group nodes as needed.

# - Root(Group)
#     - Basic Info: Name - Alex Johnson, Birthday - 1995-08-15, Phone - +1-555-1234, Email - alex.johnson@email.com, Address - 123 Maple Street, Springfield
#         - Friends: Taylor Smith (Birthday: 1994-02-20, Phone: +1-555-5678), Jordan Lee (Birthday: 1993-11-30, Phone: +1-555-9101), Morgan Brown (Birthday: 1996-05-25, Phone: +1-555-1213)
#     - Personal Details: Occupation - Software Developer, Hobbies - reading, hiking, coding, photography
#         - Work & Goals: Company - Tech Solutions Inc., Position - Front-End Developer, Work Email - alex.j@techsolutions.com, Work Phone - +1-555-4321, Goals - Learn a new programming language, Complete a marathon, Read 20 books this year
# '''
# print(f'\n\n######## Question ########\n\n{question}')
# answer,solution = store.find('MainAgent:main_agent')(question=question,debug=False)
# print(f'\n\n######## Answer ########\n\n{answer}')
# print(f'\n\n######## Solution ########\n\n{solution}')