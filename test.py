from LLMAbstractModel import *
ss = PythonDictStorage()
# llm = Model4LLMs.ChatGPT4oMini()


def test(materials=''):
    # ideas : 1.purpose 2.reference materials 3.previous outputs ==> current outputs 
    reference_materials = materials.split()
    previous_outputs = []

    llm = Model4LLMs.ChatGPT4oMini()
    # llm.build_system(purpose='...')
    def system_prompt(limit_words,code_path,code_lang,code,pre_explanation):
        return f'''You are an expert in code explanation, familiar with Electron, and Python.
I have an app built with Electron and Python.
I will provide pieces of the project code along with prior explanations.
Your task is to read each new code snippet and add new explanations accordingly.  
You should reply in Japanese with explanations only, without any additional information.

## Your Reply Format Example (should not over {limit_words} words)
```explanation
- This code shows ...
```
                           
## Code Snippet
code path : {code_path}
```{code_lang}
{code}
```

## Previous Explanations
```explanation
{pre_explanation}
```'''
    
    def messages(limit_words,code_path,code_lang,code,pre_explanation):
        return [
            {"role": "system", "content": system_prompt(limit_words,code_path,code_lang,code,pre_explanation)},
        ]

    def outputFormatter(output=''):
        def extract_explanation_block(text):
            matches = re.findall(r"```explanation\s*(.*)\s*```", text, re.DOTALL)
            return matches if matches else []
        return extract_explanation_block(output)[0]

    for m in reference_materials:
        preout = previous_outputs[-1] if len(previous_outputs)>0 else None
        msgs = messages(code=m,pre_explanation=preout,limit_words=100)
        tokens = llm.get_token_count(msgs)
        output = llm.gen(msgs)
        previous_outputs.append(outputFormatter(output))


