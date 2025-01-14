import json
import re
from pydantic import Field
from typing import Any, Callable, Optional, List

from .BasicModel import Model4Basic
from .LLMsModel import LLMsStore, Model4LLMs

descriptions = Model4LLMs.Function.param_descriptions
@descriptions('Extract text by regx pattern',
              text='input text')
class RegxExtractor(Model4LLMs.Function):
    regx:str = Field(description='regx pattern')
    is_json:bool=False
    def __call__(self,text:str):
        return self.extract(text)
    
    def extract(self,text,only_first=True)->str:
        matches = re.findall(self.regx, text, re.DOTALL)        
        if not self._try_binary_error(lambda:matches[0]):
            self._log_error(ValueError(f'cannot match {self.regx} at {text}'))
            return text
        if only_first:
            m = matches[0]
            if self.is_json:return json.loads(m)
            return m
        else:
            if self.is_json:return [json.loads(m) for m in matches]
            return matches

@descriptions('String Template for format function',
              args='string.format args')
class StringTemplate(Model4LLMs.Function):
    string:str = Field(description='string of f"..."')
    def __call__(self,*args,**kwargs):
        return self.string.format(*args,**kwargs)


@descriptions('Classification Template for performing conditional checks on a target',
              target='The object or value to be classified',
              condition_funcs='List of condition functions to evaluate the target')
class ClassificationTemplate(Model4LLMs.Function):
    def __call__(self, target: Any, condition_funcs: List[Callable[[Any], Any]] = []) -> List[bool]:
        # Initialize an empty result list
        results = []
        for idx, func in enumerate(condition_funcs):
            try:
                # Attempt to apply the function to the target and ensure it's a boolean result
                result = bool(func(target))
                results.append(result)
            except Exception as e:
                # Log the error gracefully and append False as a default failure value
                print(f"Error in condition function {idx} for target {target}: {e}")
                results.append(False)

        return results
    
store = LLMsStore()
store.add_new_obj(RegxExtractor(regx='')).get_controller().delete()
store.add_new_obj(StringTemplate(string='')).get_controller().delete()
store.add_new_obj(ClassificationTemplate()).get_controller().delete()

class TextFile(Model4Basic.AbstractObj):
    file_path: str
    chunk_lines: int = Field(default=1000, gt=0,
                 description="Number of lines per chunk, must be greater than 0")
    overlap_lines: int = Field(default=100, ge=0,
                 description="Number of overlapping lines between chunks, must be non-negative")
    current_position: int = Field(default=0, ge=0,
                 description="Current position in the file")
    line_count: Optional[int] = None
    _file:Any = None
    _current_chunk:Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._file = open(self.file_path, 'r', encoding='utf-8')  # Open the file in read mode
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
        self._file.seek(0)
        self.current_position = 0

    def read_chunk(self) -> Optional[List[str]]:
        """
        Read the next chunk of lines with overlap.

        :return: List of lines in the current chunk.
        """
        if self.current_position >= self.line_count:
            return None  # End of file

        # Calculate start and end line numbers
        start_line = max(self.current_position - self.overlap_lines, 0)
        end_line = min(self.current_position + self.chunk_lines, self.line_count)

        # Seek to the start line
        if start_line > 0:
            self._reset_file()
            for _ in range(start_line):
                self._file.readline()

        # Read the chunk
        chunk = []
        for _ in range(start_line, end_line):
            line = self._file.readline()
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
        self._current_chunk = self.read_chunk()
        return self

    def __next__(self) -> List[str]:
        """
        Return the next chunk of lines.
        """
        if not self._current_chunk:
            self._current_chunk = self.read_chunk()

        if self._current_chunk is None or len(self._current_chunk) == 0:
            raise StopIteration

        chunk = self._current_chunk
        self._current_chunk = self.read_chunk()
        return chunk

    def close(self):
        """
        Close the file when done.
        """
        self._file.close()

# Example usage:
# text_file = TextFile(file_path="example.txt")
# for chunk in text_file:
#     print(chunk)
# text_file.close()
