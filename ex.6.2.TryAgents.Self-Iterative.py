
import requests
from LLMAbstractModel.utils import RegxExtractor
from LLMAbstractModel import LLMsStore,Model4LLMs
descriptions = Model4LLMs.Function.param_descriptions
def myprint(string):
    print('##',string,':\n',eval(string),'\n')

store = LLMsStore()
vendor = store.add_new_openai_vendor(api_key="OPENAI_API_KEY")
debug = True

# https://github.com/richards199999/Self-Iterative-Agent-System-for-Complex-Problem-Solving/tree/main

## add Main model
@descriptions('Workflow function of main model workflow', question='The question to ask the LLM')
class MainAgent(Model4LLMs.Function):
    solutions:int = 2
    each_iterations:int = 2
    llm_id:str
    eval_agent_id:str
    relevantConcepts_extract:RegxExtractor = RegxExtractor(regx=r"<relevantConcepts>\s*(.*)\s*</relevantConcepts>")
    thoughts_extract:RegxExtractor = RegxExtractor(regx=r"<thoughts>\s*(.*)\s*</thoughts>")
    calculation_extract:RegxExtractor = RegxExtractor(regx=r"<calculation>\s*(.*)\s*</calculation>")
    result_extract:RegxExtractor = RegxExtractor(regx=r"<result>\s*(.*)\s*</result>")
    system_prompt:str='''You will be answering mathematical questions as a professional mathematician. When the user provide a question, follow these steps:

<step1>
Carefully read the question and identify the mathematical concepts, theorems, and knowledge points required to solve the problem. Write down these concepts inside <relevantConcepts> tags.
</step1>

<step2>
Brainstorm ideas and possible solution approaches. Write your **detailed** thoughts, insights, and potential strategies inside <thoughts> tags. Consider multiple angles and perspectives in your thinking process. You need to flexibly use formulas and theorems to assist you in thinking. You should always **think outside the box and find new ways** to find key ideas for solving problems.
</step2>

<step3>
Begin the solution process based on your thinking process. As you solve the problem, show all calculations in detail inside <calculation> tags. You MUST provide **complete formula enumeration, data substitution and detailed calculation process**. When encountering complex calculations, **break them down into the simplest possible steps** to maintain rigor. DO NOT skip or miss any part of calculation, even they are considered to be "unnecessary" or easy. Use correct formatting for mathematical expressions and equations.
</step3>

<step4>
After completing the solution, present your final result inside <result> tags. Ensure that the result is **in the correct format and includes any necessary units or labels**.
</step4>

You should use XML tags to present your complete solution, including the relevant concepts, thinking process, detailed calculations, final result, and self review, in a well-organized and easy-to-follow format.

Remember to **maintain a rigorous, professional, and clear approach** throughout the problem-solving process. Your goal is to **provide a thorough, accurate, and well-explained solution** to the provided mathematical question.

Note: Your entire process will be sent to another AI assistant for evaluation, so you should refine your process after you receive the detailed feedback.
```
'''
    final_prompt:str = '''You will be evaluating a set of solutions to a mathematical problem and determining the best solution among them. Your task is to evaluate each solution based on various criteria and select the best one. Follow these steps:

- Read through each solution carefully, paying attention to the relevant concepts, thinking process, calculations, and final result. Assess the clarity, logic, and organization of each solution.

- Assess the reasoning and logic behind each solution. Determine how well the thinking process is explained and whether it is mathematically sound. Assign a score from 1 to 5 for reasoning and logic, where 5 represents the most coherent and logical approach.

- Evaluate the calculations in each solution. Check for any errors, inconsistencies, or step incompleteness in the problem-solving process. Assign a score from 1 to 5 for these factors, where 5 represents the highest accuracy.

- Consider the clarity and presentation of each solution. Evaluate how well the solution is structured, how easy it is to follow, and whether the final result is presented in the correct format. Assign a score from 1 to 5 for clarity and presentation, where 5 represents the most clear and well-presented solution.

- Based on the scores assigned for calculation accuracy, reasoning and logic, and clarity and presentation, determine an overall score for each solution. The overall score should be a weighted average, with calculation completeness being the most important factor, followed by reasoning and logic, and then clarity and presentation.

Present your selection using the following template:
"""
Solution 1: Calculation: (score), Reasoning and Logic: (score), Clarity and Presentation: (score), Overall Score: (score)
Solution 2: Calculation: (score), Reasoning and Logic: (score), Clarity and Presentation: (score), Overall Score: (score)

The best solution is: (Solution number)
Justification: (Provide a brief explanation of why this solution was selected as the best)
"""

Remember to be objective, thorough, and consistent in your evaluation. Your goal is to identify the solution that demonstrates the highest level of calculation, logical reasoning, and clear presentation.
'''

    def change_llm(self,llm_obj:Model4LLMs.AbstractLLM):
        self.ll_id = llm_obj.get_id()
        self.get_controller().store()
    
    def __call__(self,question,debug=False):
        debugprint = lambda msg:print(f'--> [main_agent]: {msg}') if debug else lambda:None

        main_llm:Model4LLMs.AbstractLLM = self.get_controller().storage().find(self.llm_id)
        main_llm.system_prompt = self.system_prompt
        eval_agent = self.get_controller().storage().find(self.eval_agent_id)

        debugprint(f'Asking main_llm with: [{dict(question=question)}]')
        def get_review(answer):
            return eval_agent(question,
                                self.relevantConcepts_extract(answer),
                                self.thoughts_extract(answer),
                                self.calculation_extract(answer),
                                self.result_extract(answer),debug=debug)
        
        def thinking(question,result=None,review=None):
            question_tmp = '## question\n{}\n'+('## result\n{}\n' if result else'')+('## review\n{}\n' if review else'')
            ask = question_tmp.format(question,result,review)
            answer = main_llm(ask)
            result = self.result_extract(answer)
            calculation = self.calculation_extract(answer)
            review = get_review(answer)
            return question,ask,answer,calculation,result,review
        rs = []
        for ii in range(self.solutions):
            result=None
            review=None
            for i in range(self.each_iterations):
                print(f'############# ite=={i} ##############')
                question,ask,answer,calculation,result,review = thinking(question,result,review)
                print(review)
                debugprint(ask)
                # debugprint(f'answer == [{dict(answer=answer)}]')
                # debugprint(f'relevantConcepts == [{self.relevantConcepts_extract(answer)}]')
                # debugprint(f'thoughts == [{self.thoughts_extract(answer)}]')
                # debugprint(f'calculation == [{self.calculation_extract(answer)}]')
                # debugprint(f'result == [{self.result_extract(answer)}]')
            
            rs.append(result+f'\n{calculation}')
            print(f'############# solution=={ii} ##############')
            print("\n### " + rs[-1])

        main_llm.system_prompt = self.final_prompt
        rs = "\n### ".join(rs)
        
        print(f'############# final ##############')
        print(f'## question\n{question}\n## results\n### {rs}\n')
        return main_llm(f'## question\n{question}\n## results\n### {rs}\n')


@descriptions('Workflow function of eval model workflow', question='The question to ask the LLM')
class EvalAgent(Model4LLMs.Function):
    llm_id:str
    initialReview_extract:RegxExtractor = RegxExtractor(regx=r"<initialReview>\s*(.*)\s*</initialReview>")
    reasoningFeedback_extract:RegxExtractor = RegxExtractor(regx=r"<reasoningFeedback>\s*(.*)\s*</reasoningFeedback>")
    calculationErrors_extract:RegxExtractor = RegxExtractor(regx=r"<calculationErrors>\s*(.*)\s*</calculationErrors>")
    overallAssessment_extract:RegxExtractor = RegxExtractor(regx=r"<overallAssessment>\s*(.*)\s*</overallAssessment>")
    system_prompt:str='''You will be reviewing the problem-solving process of another AI assistant that has answered a mathematical question. Your task is to evaluate the solution and provide a detailed review for refinement. Follow these steps:

<step1>
Carefully read through the original question and entire solution, paying close attention to the relevant concepts, thinking process, calculations, and final result. Assess whether the solution is clear, logical, and well-organized. Write your initial review in <initialReview> tags.
</step1>

<step2>
Evaluate the reasoning and logic behind the solution. Ensure that the thinking process is clear, coherent, and mathematically sound. If you find any areas that need clarification or improvement, provide your suggestions inside <reasoningFeedback> tags.
</step2>

<step3>
Re-do the calculations presented in the <calculation> section **carefully and step-by-step** to verify the accuracy. Break down the calculations into the simplest possible steps and check each step for errors. You must not be careless and treat every part with rigor. Don't neglect checking any calculation part of the solution process. If you find any mistakes, note them down inside <calculationErrors> tags.
</step3>

<step4>
Provide an overall assessment of the solution's thoroughness, accuracy, and clarity inside <overallAssessment> tags. Highlight the strengths and weaknesses of the solution and offer suggestions for improvement, if any.
</step4>

You should use XML tags to present your complete evaluation, including initial review, calculation errors, reasoning feedback, and overall assessment, in a well-organized and easy-to-follow format.

Remember to be thorough, constructive, and professional in your review. Your goal is to help improve the quality and accuracy of the mathematical problem-solving process.
'''

    def change_llm(self,llm_obj:Model4LLMs.AbstractLLM):
        self.ll_id = llm_obj.get_id()
        self.get_controller().store()
    
    def __call__(self,question,relevantConcepts,thoughts,calculation,result,
                 debug=True):
        debugprint = lambda msg:print(f'--> [eval_agent]: {msg}') if debug else lambda:None
        question = f'## question\n{question}\n## relevantConcepts\n{relevantConcepts}\n## thoughts\n{thoughts}\n## calculation\n{calculation}\n## result\n{result}'
        eval_llm:Model4LLMs.AbstractLLM = self.get_controller().storage().find(self.llm_id)
        eval_llm.system_prompt = self.system_prompt

        debugprint(f'Asking eval_llm with: [{dict(question=question)}]')

        answer = eval_llm(question)
        # debugprint(f'initialReview == [{self.initialReview_extract(answer)}]')
        # debugprint(f'reasoningFeedback == [{self.reasoningFeedback_extract(answer)}]')
        # debugprint(f'calculationErrors == [{self.calculationErrors_extract(answer)}]')
        # debugprint(f'overallAssessment == [{self.overallAssessment_extract(answer)}]')
        return f'### initialReview\n{self.initialReview_extract(answer)}\n### reasoningFeedback\n{self.reasoningFeedback_extract(answer)}\n### calculationErrors\n{self.calculationErrors_extract(answer)}\n### overallAssessmt\n{self.overallAssessment_extract(answer)}'

main_llm = store.add_new_chatgpt4omini(vendor_id='auto')
eval_llm = store.add_new_chatgpt4omini(vendor_id='auto')

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
