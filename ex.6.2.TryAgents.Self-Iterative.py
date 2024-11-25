
from LLMAbstractModel.utils import RegxExtractor
from LLMAbstractModel import LLMsStore,Model4LLMs
descriptions = Model4LLMs.Function.param_descriptions
def myprint(string):
    print('##',string,':\n',eval(string),'\n')

store = LLMsStore()
vendor = store.add_new_openai_vendor(api_key="OPENAI_API_KEY",timeout=60)
vendor = store.add_new_Xai_vendor(api_key='XAI_API_KEY',timeout=600)
debug = True


# https://github.com/richards199999/Self-Iterative-Agent-System-for-Complex-Problem-Solving/tree/main

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
    result_extract:RegxExtractor = RegxExtractor(regx=r"<result>\s*(.*)\s*</result>")
    system_prompt: str = '''**You will be addressing questions and tasks as a professional problem-solver. When the user provides a task or question, follow these steps:**

### Step 1: Identify Relevant Concepts
Carefully read the question or task to identify the key concepts, principles, and knowledge areas required to address it. Write these concepts inside `<relevant_concepts>` tags.  

### Step 2: Brainstorm and Plan
Brainstorm ideas and possible approaches to solving the task. Write your **detailed** thoughts, insights, and potential strategies inside `<thoughts>` tags. Consider multiple perspectives and explore creative, out-of-the-box solutions. Flexibly apply relevant principles, tools, or methodologies to assist your thinking.

### Step 3: Execute the Solution  
Start executing the solution based on your brainstorming. As you work through the task, document all steps in detail inside `<process>` tags. Provide **a comprehensive breakdown of all actions, reasoning, and steps involved**, ensuring clarity and accuracy. Break down complex steps into simpler components, and DO NOT skip or omit any part, even if it seems obvious or easy. Maintain clear formatting and structure for the process.  

### Step 4: Present the Final Result  
After completing the task, present your final outcome inside `<result>` tags. Ensure the result is **clearly formatted, concise, and includes any necessary units, labels, or context**.  

### Additional Notes
Remember to **maintain a professional, thorough, and precise approach** throughout. Your goal is to **provide a well-explained, accurate, and actionable solution** for any task or question presented.

*Note:* Your entire process will be sent to another assistant for evaluation, so refine your approach based on any detailed feedback you receive.
'''
    final_prompt: str = r'''You will be evaluating a set of solutions to a problem and determining the best solution among them. Your task is to evaluate each solution based on various criteria and select the best one. Follow these steps:

- Read through each solution carefully, paying attention to the relevant concepts, thoughts, processes, and final result. Assess the clarity, logic, and organization of each solution.

- Assess the reasoning and logic behind each solution. Determine how well the thoughts are explained. Assign a score from 1 to 5 for reasoning and logic, where 5 represents the most coherent and logical approach.

- Evaluate the processes in each solution. Check for any errors, inconsistencies, or step incompleteness in the problem-solving process. Assign a score from 1 to 5 for these factors, where 5 represents the highest accuracy.

- Consider the clarity and presentation of each solution. Evaluate how well the solution is structured, how easy it is to follow, and whether the final result is presented in the correct format. Assign a score from 1 to 5 for clarity and presentation, where 5 represents the most clear and well-presented solution.

- Based on the scores assigned for process accuracy, reasoning and logic, and clarity and presentation, determine an overall score for each solution. The overall score should be a weighted average, with process completeness being the most important factor, followed by reasoning and logic, and then clarity and presentation.

Present your selection using the following template:
```template
Solution 1: process: {score:.2f}, reasoning_and_logic: {score:.2f}, clarity_and_presentation: {score:.2f}, overall_score: {score:.2f}
Solution 2: process: {score:.2f}, reasoning_and_logic: {score:.2f}, clarity_and_presentation: {score:.2f}, overall_score: {score:.2f}

The best solution is: {solution_number}
Justification: (Provide a brief explanation of why this solution was selected as the best)
```

Remember to be objective, thorough, and consistent in your evaluation. Your goal is to identify the solution that demonstrates the highest level of process, logical reasoning, and clear presentation.
'''

    def change_llm(self, llm_obj: Model4LLMs.AbstractLLM):
        self.ll_id = llm_obj.get_id()
        self.get_controller().store()
        
    def __call__(self, question, debug=False):
        debug_print = lambda msg: print(f'--> [main_agent]: {msg}') if debug else lambda: None

        main_llm: Model4LLMs.AbstractLLM = self.get_controller().storage().find(self.llm_id)
        main_llm.system_prompt = self.system_prompt
        eval_agent = self.get_controller().storage().find(self.eval_agent_id)

        debug_print(f'Asking main_llm with: [{dict(question=question)}]')
        def get_review(answer):
            return eval_agent(
                question,
                self.relevant_concepts_extract(answer),
                self.thoughts_extract(answer),
                self.process_extract(answer),
                self.result_extract(answer),
                debug=debug
            )
        
        def thinking(question, result=None, review=None):
            question_tmp = '## question\n{}\n' + (
                           '## result\n{}\n' if result else '') + (
                           '## review\n{}\n' if review else '')
            ask = question_tmp.format(question, result, review)
            answer = main_llm(ask)
            result = self.result_extract(answer)
            process = self.process_extract(answer)
            review = get_review(answer)
            return question, ask, answer, process, result, review

        rs = []
        for ii in range(self.solutions):
            result = None
            review = None
            for i in range(self.each_iterations):
                print(f'############# ite=={i} ##############')
                question, ask, answer, process, result, review = thinking(question, result, review)
                print(review)
                debug_print(ask)
                
            rs.append(f'### Solution {ii}\n' + f'\n```result\n{result}\n```' + f'\n```explain\n{process}\n```')
            print(f'############# solution=={ii} ##############')
            print("\n" + rs[-1])

        main_llm.system_prompt = self.final_prompt
        rs = "\n".join(rs)
        
        print(f'############# final ##############')
        print(f'## question\n{question}\n## results\n{rs}\n')
        return main_llm(f'## question\n{question}\n## results\n{rs}\n')


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
Carefully read through the original question and the entire solution, paying close attention to the relevant concepts, reasoning, processes, and the final result. Assess whether the solution is clear, logical, and well-organized. Write your initial review within `<initial_review>` tags.  

### step2
Evaluate the reasoning and logic behind the solution. Ensure that the reasoning is clear and coherent. If you find any areas that need clarification or improvement, provide your suggestions within `<reasoning_feedback>` tags.  

### step3
Re-do the processes presented in the `<process>` section **carefully and step-by-step** to verify their accuracy. Break the processes down into the simplest possible steps and check each step for errors. You must treat each part with rigor and avoid carelessness. Ensure that no part of the solution process is neglected during verification. If you find any mistakes, document them within `<process_errors>` tags.  

### step4
Provide an overall assessment of the solution's thoroughness, accuracy, and clarity within `<overall_assessment>` tags. Highlight the strengths and weaknesses of the solution and offer suggestions for improvement, if any.  

Use XML tags to present your complete evaluation, including initial review, process errors, reasoning feedback, and overall assessment, in a well-organized and easy-to-follow format.  

Remember to be thorough, constructive, and professional in your review. Your goal is to help improve the quality and accuracy of the problem-solving process.'''

    def change_llm(self,llm_obj:Model4LLMs.AbstractLLM):
        self.ll_id = llm_obj.get_id()
        self.get_controller().store()
    
    def __call__(self,question,relevant_concepts,thoughts,process,result,
                 debug=True):
        debugprint = lambda msg:print(f'--> [eval_agent]: {msg}') if debug else lambda:None
        question = ('## question\n{}\n' + (
                    '## relevant concepts\n{}\n' if relevant_concepts else '') + (
                    '## thoughts\n{}\n' if thoughts else '') + (
                    '## process\n{}\n' if process else '') + (
                    '## result\n{}\n' if result else '')).format(
                        question,relevant_concepts,thoughts,process,result)
                
        eval_llm:Model4LLMs.AbstractLLM = self.get_controller().storage().find(self.llm_id)
        eval_llm.system_prompt = self.system_prompt

        debugprint(f'Asking eval_llm with: [{dict(question=question)}]')

        answer = eval_llm(question)
        # debugprint(f'initial_review == [{self.initial_review_extract(answer)}]')
        # debugprint(f'reasoning_feedback == [{self.reasoning_feedback_extract(answer)}]')
        # debugprint(f'process_errors == [{self.process_errors_extract(answer)}]')
        # debugprint(f'overall_assessment == [{self.overall_assessment_extract(answer)}]')
        return f'### initial review\n{self.initial_review_extract(answer)}\n### reasoning feedback\n{self.reasoning_feedback_extract(answer)}\n### process errors\n{self.process_errors_extract(answer)}\n### overall assessmt\n{self.overall_assessment_extract(answer)}'

main_llm = store.add_new_chatgpt4omini(vendor_id='auto')
eval_llm = store.add_new_chatgpt4omini(vendor_id='auto')

# main_llm = store.add_new_grok(vendor_id='auto')
# eval_llm = store.add_new_grok(vendor_id='auto')

store.add_new_obj(EvalAgent(llm_id=eval_llm.get_id()),id='EvalAgent:eval_agent')
store.add_new_obj(MainAgent(llm_id=main_llm.get_id(),eval_agent_id='EvalAgent:eval_agent'),id='MainAgent:main_agent')

data = store.dumps()
store.clean()
store.loads(data)

# Example usage
answer = store.find('MainAgent:main_agent')(
            question='You have six horses and want to race them to see which is fastest. What is the best way to do this?',
            debug=False)


print(f'\n\n######## Answer ########\n\n{answer}')
