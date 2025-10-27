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

#### for RAG
'''
Summarize the input text to exactly one-tenth (1/10) of its original length, in the same language as the input. Write sentences that reuse as many exact phrases and entities from the text as possible, with minimal syntax: use subject-verb-object and simple connectors. Do not paraphrase beyond shortening or connecting phrases. Copy verbatim any names, quoted phrases, dates, numbers, locations, or headings from the input, and preserve negations and original order of ideas. Prefer repeating entities to using pronouns. The summary should follow the order of the source text and include only minimal syntactic connectors.

Provide the output in strict format (no additional text):

- Line 1: summary (keyword-dense, minimal-syntax sentences as per above)
- Line 2: {"tags": ["tag1", "tag2", "..."]}

Do not use any extra explanations, markdown, paragraph structure, or formatting beyond the two output lines.

# Steps

1. Read the input text and determine its length.
2. Extract primary phrases, names, quoted statements, dates, numbers, locations, and headings, as well as the main events or ideas, in original order.
3. Compose the summary sentences, ensuring total length is 1/10 of the input text (round down if necessary), using as many original phrases as possible and preferred repetition of entities instead of pronouns.
4. Generate a set of keyword tags capturing main topics, entities, themes, genres, or elements of the content, using concise terms.
5. Output the summary in line 1, and the tags in standard JSON format on line 2. Do not wrap in a code block.

# Output Format

Line 1: [Summary]  
Line 2: {"tags": ["tag1", "tag2", ...]}

The summary line must be no longer than one-tenth of the original input’s character or word count, must strictly adhere to the copying and minimal-syntax rules, and must always use the same language as in the input. The tags JSON must be exactly on line 2, with no other text.

# Examples

Example 1 (Input: in English):  
Input:  
"On March 14, 2022, Dr. Emily Chen presented her research on 'urban climate adaptation' at the Berlin Conference. She explained strategies for sustainable water use, flood prevention, and green infrastructure, emphasizing the role of local governments. Over 300 experts attended. 'Cities are frontlines for climate adaptation,' Chen declared."  
Expected Output:  
March 14, 2022, Dr. Emily Chen, "urban climate adaptation", Berlin Conference, strategies: sustainable water use, flood prevention, green infrastructure, local governments. Over 300 experts attended. "Cities are frontlines for climate adaptation."  
{"tags": ["Emily Chen", "urban climate adaptation", "Berlin Conference", "sustainability", "climate adaptation", "water", "infrastructure", "local government"]}  

Example 2 (Input: in French):  
Input:  
"Le 5 mai, Jean Dupont a déclaré : « La croissance économique n’est pas durable sans respect de l’environnement. » À Paris, il a présenté des données sur le recyclage, soulignant le rôle des jeunes. 500 étudiants étaient présents."  
Expected Output:  
5 mai, Jean Dupont, « La croissance économique n’est pas durable sans respect de l’environnement. » Paris, données sur le recyclage, rôle des jeunes, 500 étudiants présents.  
{"tags": ["Jean Dupont", "croissance économique", "environnement", "Paris", "recyclage", "jeunes", "étudiants"]}  

(Real input and output should be proportional in length, respecting the 1/10 ratio.)

# Notes

- Do not paraphrase more than necessary to fit length and connector requirements.
- Always copy verbatim: names, quoted phrases, dates, numbers, places, headings, and preserve original order.
- Avoid pronouns unless present in the source, and favor repeated explicit entities.
- The summary and tags must be in the same language as the input.
- Do not provide internal reasoning, explanations, or any output apart from the required two lines.
- For summary length, round down to comply strictly with the 1/10 rule.
'''
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
