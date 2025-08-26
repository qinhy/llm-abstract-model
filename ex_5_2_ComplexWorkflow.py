from LLMAbstractModel.utils import Base64ToStringDecoder, StringTemplate, RegxExtractor, StringToBase64Encoder
from LLMAbstractModel import LLMsStore,Model4LLMs

def myprint(string):
    print('##',string,':\n',eval(string),'\n')

def init(store = LLMsStore()):    
    store.add_new_vendor(Model4LLMs.OpenAIVendor)(api_key='OPENAI_API_KEY')
    store.add_new_vendor(Model4LLMs.DeepSeekVendor)(api_key='DEEPSEEK_API_KEY')
    new_llm = lambda system_prompt: store.add_new_llm( Model4LLMs.ChatGPTDynamic)(
        llm_model_name='gpt-5-nano',vendor_id='auto',limit_output_tokens = 2048,system_prompt=system_prompt)
    solver = new_llm(
    system_prompt='''You will act as a professional problem-solver. Follow these 4 steps for any task or question without any tool ( python ... ):  

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
    
    reviewer = new_llm(system_prompt='''You will review another assistant's solution to a question. Your task is to evaluate and improve it by following these 4 steps:

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
    
    jugde = new_llm(system_prompt='''Task: Evaluate multiple solutions to a problem and select the best one. Follow these steps:

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
    
    b642str: Base64ToStringDecoder = store.add_new_obj(Base64ToStringDecoder())

    question_tmplt:StringTemplate = store.add_new_obj(StringTemplate(para=dict(string=
        'Please provide a better answer to the following question based on the previous process and review.\n\nQuestion:\n{qu}\n\nPrevious process:\n{pp}\n\nPrevious solution review:\n{ps}')))
    review_tmp:StringTemplate = store.add_new_obj(StringTemplate(para=dict(string=                                                                           
        'Question\n{qu}\n\nRelevant concepts\n{rc}\n\nThoughts\n{th}\n\nProcess\n{pr}\n\nSolution\n{sl}')))
    
    store.add_new_obj(
        Model4LLMs.MermaidWorkflow(
            mermaid_text=f'''
    graph TD
        {b642str.get_id()}["{{'args': {{'encoded': 'THE_QUESTION'}} }}"]

        {b642str.get_id()} -- "{{'plain':'messages'}}" --> {solver.get_id()}
        {solver.get_id()} -- "{{'data':'text'}}" --> {relevant_concepts_extract.get_id()}
        {solver.get_id()} -- "{{'data':'text'}}" --> {thoughts_extract.get_id()}
        {solver.get_id()} -- "{{'data':'text'}}" --> {process_extract.get_id()}
        {solver.get_id()} -- "{{'data':'text'}}" --> {solution_extract.get_id()}

        {b642str.get_id()} -- "{{'plain':'qu'}}" --> {review_tmp.get_id()}
        {relevant_concepts_extract.get_id()} -- "{{'data':'rc'}}" --> {review_tmp.get_id()}
        {thoughts_extract.get_id()} -- "{{'data':'th'}}" --> {review_tmp.get_id()}
        {process_extract.get_id()} -- "{{'data':'pr'}}" --> {review_tmp.get_id()}
        {solution_extract.get_id()} -- "{{'data':'sl'}}" --> {review_tmp.get_id()}
        {review_tmp.get_id()} -- "{{'data':'messages'}}" --> {reviewer.get_id()}

        {reviewer.get_id()} -- "{{'data':'text'}}" --> {initial_review_extract.get_id()}
        {reviewer.get_id()} -- "{{'data':'text'}}" --> {reasoning_feedback_extract.get_id()}
        {reviewer.get_id()} -- "{{'data':'text'}}" --> {process_errors_extract.get_id()}
        {reviewer.get_id()} -- "{{'data':'text'}}" --> {overall_assessment_extract.get_id()}

        {b642str.get_id()} -- "{{'plain':'qu'}}" --> {question_tmplt.get_id()}
        {process_extract.get_id()} -- "{{'data':'pp'}}" --> {question_tmplt.get_id()}
        {reviewer.get_id()} -- "{{'data':'ps'}}" --> {question_tmplt.get_id()}

    '''),id='solveAndReview')
    
    return store    

store = init()
data = store.dumps()

store.clean()
store.loads(data)

solve_and_review:Model4LLMs.MermaidWorkflow = store.find('solveAndReview')
solve_and_review.b64_placeholder('How many characters of the letter "r" in "rarspberrrry"?','THE_QUESTION')
for i in range(3):
    res = solve_and_review()
    solve_and_review.b64_placeholder(res,'THE_QUESTION')
    qu = res
print('######## final answer #######')
print(res)