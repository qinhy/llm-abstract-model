
import json
from LLMAbstractModel.utils import StringTemplate, RegxExtractor
from LLMAbstractModel import LLMsStore,Model4LLMs
descriptions = Model4LLMs.Function.param_descriptions
def myprint(string):
    print('##',string,':\n',eval(string),'\n')


def init(store = LLMsStore()):    
    vendor = store.add_new_openai_vendor(api_key='OPENAI_API_KEY')
    solver = store.add_new_chatgpt4omini(vendor_id='auto',limit_output_tokens = 2048,
                        system_prompt='''You will act as a professional problem-solver. Follow these steps for any task or question:  

Step 1: Identify Key Concepts  
Read the task carefully and identify the main ideas and knowledge areas needed. Use `<relevant_concepts>` xml tags for this.

Step 2: Plan and Brainstorm  
Think of ideas and approaches to solve the task, exploring different strategies and perspectives. Document your thoughts within `<thoughts>` xml tags in this format:  
- Observation: Note details from the user's input or previous steps.
- Thought: Consider the observation and decide on the next step.
- Action: Specify what action to take next.

Step 3: Solve the Problem  
Execute the solution based on your plan. Provide a detailed step-by-step explanation within `<process>` xml tags, breaking down complex ideas into simpler parts.

Step 4: Present the Solution  
Share the final outcome clearly and concisely in `<solution>` xml tags. Include necessary labels, units, and context.

Additional Notes:  
Stay professional, thorough, and precise. Your solution should be clear, actionable, and well-explained.''')
    
    reviewer = store.add_new_chatgpt4omini(vendor_id='auto',limit_output_tokens = 2048,
                        system_prompt='''You will review another assistant's solution to a question. Your task is to evaluate and improve it by following these steps:

Step 1: Review the Solution  
Read the question and solution carefully. Assess if it's clear, logical, and well-organized. Write your review inside `<initial_review>` xml tags.

Step 2: Check Reasoning  
Examine the reasoning behind the solution. Ensure it's clear and coherent. Suggest improvements, if needed, inside `<reasoning_feedback>` xml tags.

Step 3: Verify the Process  
Re-check the solution’s steps one by one. Be thorough and look for any errors. Document mistakes inside `<process_errors>` xml tags.

Step 4: Provide an Overall Assessment  
Evaluate the solution’s accuracy, clarity, and thoroughness. Highlight strengths, weaknesses, and suggestions inside `<overall_assessment>` xml tags.

Be professional and constructive to improve the solution quality.''')
    
    jugde = store.add_new_chatgpt4omini(vendor_id='auto',limit_output_tokens = 2048,
                        system_prompt='''Task: Evaluate multiple solutions to a problem and select the best one. Follow these steps:

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
   ```
   ## Solution X
   process: {score:.2f}, reasoning_and_logic: {score:.2f}, clarity_and_presentation: {score:.2f}, overall_score: {score:.2f}
   The best solution is: {solution_number}
   Justification: (Brief explanation)
   Final solution: <solution>
   ```

Key: Be objective, consistent, and thorough. Choose the solution with the best reasoning, accuracy, and clarity.''')
    
    
    relevant_concepts_extract:RegxExtractor = store.add_new_function(RegxExtractor(regx=r"<relevant_concepts>\s*(.*)\s*</relevant_concepts>"))
    thoughts_extract:RegxExtractor = store.add_new_function(RegxExtractor(regx=r"<thoughts>\s*(.*)\s*</thoughts>"))
    process_extract:RegxExtractor = store.add_new_function(RegxExtractor(regx=r"<process>\s*(.*)\s*</process>"))
    solution_extract:RegxExtractor = store.add_new_function(RegxExtractor(regx=r"<solution>\s*(.*)\s*</solution>"))
    
    initial_review_extract: RegxExtractor = store.add_new_function(RegxExtractor(regx=r"<initial_review>\s*(.*)\s*</initial_review>"))
    reasoning_feedback_extract: RegxExtractor = store.add_new_function(RegxExtractor(regx=r"<reasoning_feedback>\s*(.*)\s*</reasoning_feedback>"))
    process_errors_extract: RegxExtractor = store.add_new_function(RegxExtractor(regx=r"<process_errors>\s*(.*)\s*</process_errors>"))
    overall_assessment_extract: RegxExtractor = store.add_new_function(RegxExtractor(regx=r"<overall_assessment>\s*(.*)\s*</overall_assessment>"))
    
    question_plain:StringTemplate = store.add_new_function(StringTemplate(string='{}'))
    question_tmp:StringTemplate = store.add_new_function(StringTemplate(string=
        '"Please provide a better answer to the following question based on the previous process and review.\n\nQuestion\n{}\n\nPrevious process\n{}\n\nPrevious solution review\n{}'))
    review_tmp = store.add_new_function(StringTemplate(string=
        'Question\n{}\n\nRelevant concepts\n{}\n\nThoughts\n{}\n\nProcess\n{}\n\nSolution\n{}'))
    
    review_collect = store.add_new_function(StringTemplate(string='{}\n{}\n{}\n{}'))
    result_collect = store.add_new_function(StringTemplate(string='{}\n\n\n\n\n{}\n\n\n\n\n{}\n\n\n\n\n{}'))    
    
    workflow = store.add_new_workflow(
        tasks={
            question_plain.get_id():['question_param'],

            question_tmp.get_id():['question_param'],
            solver.get_id():[question_tmp.get_id()],

            relevant_concepts_extract.get_id():[solver.get_id()],
            thoughts_extract.get_id():[solver.get_id()],
            process_extract.get_id():[solver.get_id()],
            solution_extract.get_id():[solver.get_id()],

            review_tmp.get_id():[question_plain.get_id(),relevant_concepts_extract.get_id(),
                                thoughts_extract.get_id(),process_extract.get_id(),solution_extract.get_id()],
            
            reviewer.get_id():[review_tmp.get_id()],

            initial_review_extract.get_id():[reviewer.get_id()],
            reasoning_feedback_extract.get_id():[reviewer.get_id()],
            process_errors_extract.get_id():[reviewer.get_id()],
            overall_assessment_extract.get_id():[reviewer.get_id()],

            review_collect.get_id():[initial_review_extract.get_id(),reasoning_feedback_extract.get_id(),
                                     process_errors_extract.get_id(),overall_assessment_extract.get_id()],

            result_collect.get_id():[question_plain.get_id(),process_extract.get_id(),
                                     review_collect.get_id(),solution_extract.get_id()],
        },
    # metadata={'tags': [str(accs[0].account_id)]}
    )

    # workflowf = lambda question:workflow(
    #     question_param=[(question,'no solution','no review'),dict()]
    # )

    q,p,r,s = 'How many characters of the letter "r" in "rarspberrrry"?','no process','no review','no solution'
    for i in range(5):
        print('#####################',i)
        workflow(question_param=[(q,s,r),dict()])
        q,p,r,s = workflow.results['final'].split('\n\n\n\n\n')
        print(s)
        print(p)
    return workflow
