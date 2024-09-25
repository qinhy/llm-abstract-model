import os
import re
from LLMAbstractModel import Model4LLMs, LLMsStore
store = LLMsStore()

vendor = store.add_new_openai_vendor(api_key=os.environ.get('OPENAI_API_KEY','null'))
chatgpt4omini = store.add_new_chatgpt4omini(vendor_id=vendor.get_id(),system_prompt='You are an expert in text summary.')

vendor = store.add_new_ollama_vendor()
gemma2 = store.add_new_gemma2(vendor_id=vendor.get_id(),system_prompt='You are an expert in text summary.')

class TextFile:
    def __init__(self, file_path, chunk_size=1000, overlap_size=100):
        """
        Initialize the TextFile object.

        :param file_path: Path to the text file.
        :param chunk_size: Number of lines per chunk.
        :param overlap_size: Number of overlapping lines between chunks.
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.current_position = 0
        self.current_chunk = None
        self.file = open(self.file_path, 'r', encoding='utf-8')  # Open the file in read mode
        self.line_count = 0
        self._calculate_total_lines()

    def _calculate_total_lines(self):
        """
        Calculate the total number of lines in the file.
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                self.line_count += 1

    def _reset_file(self):
        """
        Reset the file pointer to the beginning of the file.
        """
        self.file.seek(0)
        self.current_position = 0

    def read_chunk(self):
        """
        Read the next chunk of lines with overlap.

        :return: List of lines in the current chunk.
        """
        if self.current_position >= self.line_count:
            return None  # End of file

        # Calculate start and end line numbers
        start_line = max(self.current_position - self.overlap_size, 0)
        end_line = min(self.current_position + self.chunk_size, self.line_count)

        # Seek to the start line
        if start_line > 0:
            self._reset_file()
            for _ in range(start_line):
                self.file.readline()

        # Read the chunk
        chunk = []
        for _ in range(start_line, end_line):
            line = self.file.readline()
            if not line:
                break
            chunk.append(line)

        self.current_position = end_line

        return chunk

    def __iter__(self):
        """
        Reset the file and the position to enable iteration over the file chunks.
        """
        self._reset_file()
        self.current_chunk = self.read_chunk()
        return self

    def __next__(self):
        """
        Return the next chunk of lines.
        """
        if not self.current_chunk:
            self.current_chunk = self.read_chunk()

        if self.current_chunk is None or len(self.current_chunk) == 0:
            raise StopIteration

        chunk = self.current_chunk
        self.current_chunk = self.read_chunk()
        return chunk

    def close(self):
        """
        Close the file when done.
        """
        self.file.close()


# # Usage example:
# # Create an instance of TextFile
# text_file = TextFile('The Adventures of Sherlock Holmes.txt', chunk_size=10, overlap_size=3)

# # Read chunks using iteration
# for i,chunk in enumerate(text_file):
#     print('\n'.join(chunk))  # Do something with the chunk
#     print('----------------------------------------------')
#     if i==1:
#         break

# # Close the file
# text_file.close()

def test_summary(llm = gemma2,
                 f='The Adventures of Sherlock Holmes.txt',limit_words=1000):
    
    user_message = '''I will provide pieces of the text along with prior summarizations.
Your task is to read each new text snippet and add new summarizations accordingly.  
You should reply in Japanese with summarizations only, without any additional information.

## Your Reply Format Example (should not over {limit_words} words)
```summarization
- This text shows ...
```
                        
## Text Snippet
```text
{text}
```

## Previous Summarizations
```summarization
{pre_summarization}
```'''

    def outputFormatter(output=''):
        def extract_explanation_block(text):
            matches = re.findall(r"```summarization\s*(.*)\s*```", text, re.DOTALL)
            return matches if matches else []
        return extract_explanation_block(output)[0]
    
    previous_outputs = []    
    text_file = TextFile(f, chunk_size=100, overlap_size=30)
    for i,chunk in enumerate(text_file):
        # yield chunk        
        pre_summarization = previous_outputs[-1] if len(previous_outputs)>0 else None
        msg = user_message.format(limit_words=limit_words,text='\n'.join(chunk),
                                  pre_summarization=pre_summarization)
        # yield msg
        output = llm.gen(msg)
        output = outputFormatter(output)
        previous_outputs.append(output)
        yield output


# for i,p in enumerate(['file1','file2','file3','filen']):
#     ts = test_summary(p)
#     res = ''
#     for s in ts:
#         with open(p.replace('file','output'),'a') as f:
#             f.write(s)
#             res += s
#     with open(f'allinone','a') as f:
#         f.write(f'## {p}\n')
#         f.write(res)