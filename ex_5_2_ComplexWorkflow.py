
import json
from LLMAbstractModel.utils import StringTemplate, RegxExtractor
from LLMAbstractModel import LLMsStore,Model4LLMs
def myprint(string):
    print('##',string,':\n',eval(string),'\n')

def init(store = LLMsStore()):    
    store.add_new_vendor(Model4LLMs.OpenAIVendor)(api_key='OPENAI_API_KEY')
    store.add_new_vendor(Model4LLMs.DeepSeekVendor)(api_key='DEEPSEEK_API_KEY')
    llm_type = Model4LLMs.ChatGPT41Nano
    solver = store.add_new_llm(llm_type)(vendor_id='auto',limit_output_tokens = 2048,system_prompt='''You will act as a professional problem-solver. Follow these 4 steps for any task or question without any tool ( python ... ):  

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
    
    reviewer = store.add_new_llm(llm_type)(vendor_id='auto',limit_output_tokens = 2048,system_prompt='''You will review another assistant's solution to a question. Your task is to evaluate and improve it by following these 4 steps:

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
    
    jugde = store.add_new_llm(llm_type)(vendor_id='auto',limit_output_tokens = 2048,system_prompt='''Task: Evaluate multiple solutions to a problem and select the best one. Follow these steps:

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
    
    relevant_concepts_extract:RegxExtractor = store.add_new_obj(RegxExtractor(para=dict(regx=r"Step 1:.+?\n(.*?)(?=Step\s\d+:|$)")))
    thoughts_extract:RegxExtractor = store.add_new_obj(RegxExtractor(para=dict(regx=r"Step 2:.+?\n(.*?)(?=Step\s\d+:|$)")))
    process_extract:RegxExtractor = store.add_new_obj(RegxExtractor(para=dict(regx=r"Step 3:.+?\n(.*?)(?=Step\s\d+:|$)")))
    solution_extract:RegxExtractor = store.add_new_obj(RegxExtractor(para=dict(regx=r"Step 4:.+?\n(.*?)(?=Step\s\d+:|$)")))
    
    initial_review_extract: RegxExtractor = store.add_new_obj(RegxExtractor(para=dict(regx=r"Step 1:.+?\n(.*?)(?=Step\s\d+:|$)")))
    reasoning_feedback_extract: RegxExtractor = store.add_new_obj(RegxExtractor(para=dict(regx=r"Step 2:.+?\n(.*?)(?=Step\s\d+:|$)")))
    process_errors_extract: RegxExtractor = store.add_new_obj(RegxExtractor(para=dict(regx=r"Step 3:.+?\n(.*?)(?=Step\s\d+:|$)")))
    overall_assessment_extract: RegxExtractor = store.add_new_obj(RegxExtractor(para=dict(regx=r"Step 4:.+?\n(.*?)(?=Step\s\d+:|$)")))
    
    question_plain:StringTemplate = store.add_new_obj(StringTemplate(para=dict(string='{text}')))
    question_tmplt:StringTemplate = store.add_new_obj(StringTemplate(para=dict(string=
        'Please provide a better answer to the following question based on the previous process and review.\n\nQuestion:\n{qu}\n\nPrevious process:\n{pp}\n\nPrevious solution review:\n{ps}')))
    review_tmp:StringTemplate = store.add_new_obj(StringTemplate(para=dict(string=                                                                           
        'Question\n{qu}\n\nRelevant concepts\n{rc}\n\nThoughts\n{th}\n\nProcess\n{pr}\n\nSolution\n{sl}')))
    
    solve_and_review_1:Model4LLMs.MermaidWorkflow = store.add_new_obj(
        Model4LLMs.MermaidWorkflow(
            mermaid_text=f'''
    graph TD
        {question_plain.get_id()} -- "{{'data':'messages'}}" --> {solver.get_id()}

        {solver.get_id()} -- "{{'data':'text'}}" --> {relevant_concepts_extract.get_id()}
        {solver.get_id()} -- "{{'data':'text'}}" --> {thoughts_extract.get_id()}
        {solver.get_id()} -- "{{'data':'text'}}" --> {process_extract.get_id()}
        {solver.get_id()} -- "{{'data':'text'}}" --> {solution_extract.get_id()}

        {question_plain.get_id()} -- "{{'data':'qu'}}" --> {review_tmp.get_id()}
        {relevant_concepts_extract.get_id()} -- "{{'data':'rc'}}" --> {review_tmp.get_id()}
        {thoughts_extract.get_id()} -- "{{'data':'th'}}" --> {review_tmp.get_id()}
        {process_extract.get_id()} -- "{{'data':'pr'}}" --> {review_tmp.get_id()}
        {solution_extract.get_id()} -- "{{'data':'sl'}}" --> {review_tmp.get_id()}

        {review_tmp.get_id()} -- "{{'data':'messages'}}" --> {reviewer.get_id()}
    '''),id='solveAndReview1')
    solve_and_review_1.parse_mermaid()
    solve_and_review_1.build(text='')
    solve_and_review_1.controller.update(builds=solve_and_review_1.builds)
    # print(solve_and_review_1(text='What is Apple?')['data'])

    solve_and_review_2:Model4LLMs.MermaidWorkflow = store.add_new_obj(
        Model4LLMs.MermaidWorkflow(
            mermaid_text=f'''
    graph TD
        {solve_and_review_1.get_id()} -- "{{'data':'text'}}" --> {question_plain.get_id()}

    '''),id='solveAndReview2')
    solve_and_review_2.parse_mermaid()
    solve_and_review_2.build(text='What is Apple?')
    print(solve_and_review_2(text='What is Apple?')['data'])
    # print(solve_and_review_1.run(text='What is Apple?')['data'])
    # print(solve_and_review_2.run(text='What is Apple?'))

    # solve_and_review_2 = store.add_new_obj(solve_and_review_1.model_copy())
    # solve_and_review_3 = store.add_new_obj(solve_and_review_1.model_copy())

#     store.add_new_workflow(
#         tasks=[
#             solve_and_review_1.get_id(),
#             solve_and_review_2.get_id(),
#             solve_and_review_3.get_id(),
#         ],id='WorkFlow:solve_and_review_3_times')
    
    return store    

store = init()
# data = store.dumps()

# store.clean()
# store.loads(data)

# res = store.find('WorkFlow:solve_and_review_3_times'
#            )('How many characters of the letter "r" in "rarspberrrry"?','no process','no review','no solution')
# print(res)