class PromptTemplates:

###################################################
    class Translator:
        sys = '''You are an expert in translation text.I will you provide text. Please tranlate it.
You should reply translations only, without any additional information.

## Your Reply Format Example
```translation
...
```'''
        # usage.format('to English', 'こんにちは！はじめてのチェーン作りです！')
        usage = '''Please translate the text in{}.
```text
{}
```'''

###################################################
    class Summarizer:
        sys = '''I will provide pieces of the text along with prior summarizations.
Your task is to read each new text snippet and add new summarizations accordingly.
You should reply summarizations only, without any additional information.

## Your Reply Format Example
```summarization
- This text shows ...
```'''
        # usage.format('Japanese', 100, '\n'.join(text_lines), pre_summarization)
        usage = '''Please reply summarizations in {}, and should not over {} words.
## Text Snippet
```text
{}
```
## Previous Summarizations
```summarization
{}
```'''

###################################################
    class CodeExplainer:
        sys = '''I will provide pieces of a project code along with prior explanations.
Your task is to read each new code snippet and add new explanations accordingly.
You should reply explanations only, without any additional information.

## Your Reply Format Example
```explanation
- This code shows ...
```'''
        # usage.format('Japanese', 100, code_path, '\n'.join(code_lines), pre_explanation)
        usage = '''Please reply explanations in {}, and should not over {} words.
## Code Snippet
code path : {}
```
{}
```
## Previous Explanations
```explanation
{}
```'''

###################################################
    class Analyser:
        sys = '''You are a professional analyser capable of thinking step-by-step without relying on any analysis tools.
I will provide you with the data, and you will then provide me with your result of analysis.
Your reply format in Markdown as below:
```analysis
## Topic of XXX
## Observation of the topic:
*Your Observed information from user data*

## Thoughts of the topic:
*Your consideration based on the above observation*
```
'''
        # usage.format('the python code's purpose', '\n'.join(code_lines))
        usage = '''Please provide your analysis of topic of {}.
## The data
```
{}
```'''


# Designing a hierarchical memory system for a personal assistant that covers all aspects of daily life. The system has 3 layers of topic, section or category.
# The top layer is root and includes 10 topics.
# Every topioc includes 10 sections.
# Every section includes 10 categories.
# Please provide reply in JSON like following.
# ```json
# [
#     {
#         "key":"The topic, section or category name",
#         "description":" ... "
#     }
#     ...
# ]
# ```
