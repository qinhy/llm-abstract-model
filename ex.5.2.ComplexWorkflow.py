
import json
from LLMAbstractModel.utils import StringTemplate, RegxExtractor
from LLMAbstractModel import LLMsStore,Model4LLMs
descriptions = Model4LLMs.Function.param_descriptions
def myprint(string):
    print('##',string,':\n',eval(string),'\n')

def addllm(store:LLMsStore,system_prompt):
    return store.add_new_chatgpt4omini(vendor_id='auto',limit_output_tokens = 2048,system_prompt=system_prompt)
    # return store.add_new_deepseek(vendor_id='auto',limit_output_tokens = 2048,system_prompt=system_prompt)

def init(store = LLMsStore()):    
    store.add_new_openai_vendor(api_key='OPENAI_API_KEY')
    store.add_new_deepseek_vendor(api_key='DEEPSEEK_API_KEY')
    solver = addllm(store,system_prompt='''You will act as a professional problem-solver. Follow these 4 steps for any task or question:  

Step 1: Identify Key Concepts  
Read the task carefully and identify the main ideas and knowledge areas needed.

Step 2: Plan and Brainstorm  
Think of ideas and approaches to solve the task, exploring different strategies and perspectives. Document your thoughts in this format:
- Observation: Note details from the user's input or previous steps.
- Thought: Consider the observation and decide on the next step.
- Action: Specify what action to take next.

Step 3: Solve the Problem  
Execute the solution based on your plan. Provide a detailed step-by-step explanation, breaking down complex ideas into simpler parts.

Step 4: Present the Solution  
Share the final outcome clearly and concisely. Include necessary labels, units, and context.

Additional Notes:  
Stay professional, thorough, and precise. Your solution should be clear, actionable, and well-explained.''')
    
    reviewer = addllm(store,system_prompt='''You will review another assistant's solution to a question. Your task is to evaluate and improve it by following these 4 steps:

Step 1: Review the Solution  
Read the question and solution carefully. Assess if it's clear, logical, and well-organized.

Step 2: Check Reasoning  
Examine the reasoning behind the solution. Ensure it's clear and coherent. Suggest improvements, if needed.

Step 3: Verify the Process  
Re-check the solution steps one by one. Be thorough and look for any errors and document mistakes.

Step 4: Provide an Overall Assessment  
Evaluate the solution accuracy, clarity, and thoroughness. Highlight strengths, weaknesses, and suggestions.

Additional Notes:
Be professional and constructive to improve the solution quality.''')
    
    jugde = addllm(store,system_prompt='''Task: Evaluate multiple solutions to a problem and select the best one. Follow these steps:

1. Review Each Solution: Analyze the concepts, logic, and structure of each solution.
   - Assign scores (1-5) for:
     - Reasoning & Logic: Coherence and explanation.
     - Process Accuracy: Completeness, correctness, and consistency.
     - Clarity & Presentation: Ease of understanding and proper formatting.

2. Determine Overall Score:
   - Use a weighted average:
     - Process Accuracy: Most important.
     - Reasoning & Logic: Secondary.
     - Clarity & Presentation: Tertiary.

3. Report Results: Use this format:
   ```plaintext
   Solution X
   process: {score:.2f}, reasoning_and_logic: {score:.2f}, clarity_and_presentation: {score:.2f}, overall_score: {score:.2f}
   ...

   The best solution is: {solution_number}
   Justification: (Brief explanation)
   Final solution: (the solution)
   ```

Key: Be objective, consistent, and thorough. Choose the solution with the best reasoning, accuracy, and clarity.''')
    
    relevant_concepts_extract:RegxExtractor = store.add_new_function(RegxExtractor(regx=r"Step 1:.+?\n(.*?)(?=Step\s\d+:|$)"))
    thoughts_extract:RegxExtractor = store.add_new_function(RegxExtractor(regx=r"Step 2:.+?\n(.*?)(?=Step\s\d+:|$)"))
    process_extract:RegxExtractor = store.add_new_function(RegxExtractor(regx=r"Step 3:.+?\n(.*?)(?=Step\s\d+:|$)"))
    solution_extract:RegxExtractor = store.add_new_function(RegxExtractor(regx=r"Step 4:.+?\n(.*?)(?=Step\s\d+:|$)"))
    
    initial_review_extract: RegxExtractor = store.add_new_function(RegxExtractor(regx=r"Step 1:.+?\n(.*?)(?=Step\s\d+:|$)"))
    reasoning_feedback_extract: RegxExtractor = store.add_new_function(RegxExtractor(regx=r"Step 2:.+?\n(.*?)(?=Step\s\d+:|$)"))
    process_errors_extract: RegxExtractor = store.add_new_function(RegxExtractor(regx=r"Step 3:.+?\n(.*?)(?=Step\s\d+:|$)"))
    overall_assessment_extract: RegxExtractor = store.add_new_function(RegxExtractor(regx=r"Step 4:.+?\n(.*?)(?=Step\s\d+:|$)"))
    
    question_plain:StringTemplate = store.add_new_function(StringTemplate(string='{}'))
    question_tmp:StringTemplate = store.add_new_function(StringTemplate(string=
        'Please provide a better answer to the following question based on the previous process and review.\n\nQuestion:\n{}\n\nPrevious process:\n{}\n\nPrevious solution review:\n{}'))
    review_tmp = store.add_new_function(StringTemplate(string=
        'Question\n{}\n\nRelevant concepts\n{}\n\nThoughts\n{}\n\nProcess\n{}\n\nSolution\n{}'))
        
    solve_and_review_1 = store.add_new_workflow(
        tasks={
            question_plain.get_id():['__input__'],

            question_tmp.get_id()  :['__input__'],
            solver.get_id()        :[question_tmp.get_id()],

            relevant_concepts_extract.get_id() :[solver.get_id()],
            thoughts_extract.get_id()          :[solver.get_id()],
            process_extract.get_id()           :[solver.get_id()],
            solution_extract.get_id()          :[solver.get_id()],

            review_tmp.get_id():[question_plain.get_id(),relevant_concepts_extract.get_id(),
                                thoughts_extract.get_id(),process_extract.get_id(),solution_extract.get_id()],
            
            reviewer.get_id():[review_tmp.get_id()],

            # initial_review_extract.get_id():[reviewer.get_id()],
            # reasoning_feedback_extract.get_id():[reviewer.get_id()],
            # process_errors_extract.get_id():[reviewer.get_id()],
            # overall_assessment_extract.get_id():[reviewer.get_id()],

            'final':[question_plain.get_id(),process_extract.get_id(),
                     reviewer.get_id(),solution_extract.get_id()],
        }
    )

    solve_and_review_2 = store.add_new_obj(solve_and_review_1.model_copy(update={'_id':None}))
    solve_and_review_3 = store.add_new_obj(solve_and_review_1.model_copy(update={'_id':None}))

    store.add_new_workflow(
        tasks={
            solve_and_review_1.get_id():['__input__'],
            # solve_and_review_2.get_id():[solve_and_review_1.get_id()],
            # solve_and_review_3.get_id():[solve_and_review_2.get_id()],
        },id='WorkFlow:solve_and_review_3_times')
    
    return store    

store = init()
data = store.dumps()

store.clean()
store.loads(data)

store.find('WorkFlow:solve_and_review_3_times'
           )('How many characters of the letter "r" in "rarspberrrry"?','no process','no review','no solution')