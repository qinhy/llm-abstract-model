import base64
import json
import math
import os
import re
from types import MethodType
from pydantic import BaseModel, ConfigDict, Field, create_model
from typing import Any, Callable, Optional, List, Tuple
import tiktoken
from typing_extensions import LiteralString

from .BasicModel import Model4Basic
from .LLMsModel import LLMsStore, Model4LLMs

class RegxExtractor(Model4LLMs.MermaidWorkflowFunction):    
    description:str = Field('Extract text by regx pattern')
    
    class Parameter(BaseModel):            
        regx:str = Field(..., description='regx pattern')
        is_json:bool=False

    class Arguments(BaseModel):
        text:str = ''

    class Returness(BaseModel):
        data: Optional[str|dict|list[dict]] = None

    para: Parameter
    args: Arguments = Arguments()
    rets: Returness = Returness()
            
    def __call__(self,text:str):
        self.args.text = text
        self.rets.data = self.extract(self.args.text)
        return self.rets.data
    
    def extract(self,text,only_first=True)->str:
        matches = re.findall(self.para.regx, text, re.DOTALL)        
        if not self._try_binary_error(lambda:matches[0]):
            self._log_error(ValueError(f'cannot match {self.para.regx} at {text}'))
            return text
        if only_first:
            m = matches[0]
            if self.para.is_json:return json.loads(m)
            return m
        else:
            if self.para.is_json:return [json.loads(m) for m in matches]
            return matches

class StringTemplate(Model4LLMs.MermaidWorkflowFunction): 
    description:str = Field('Extract text by regx pattern')
    
    class Parameter(BaseModel):
        string:str = Field(description='string of f"..."')
        
    class Returness(BaseModel):
        data: str = ''

    para: Parameter
    rets: Returness = Returness()
                
    def __call__(self, *args, **kwargs):
        self.rets.data = self.para.string.format(*args,**kwargs)
        return self.rets.data

class StringToBase64Encoder(Model4LLMs.MermaidWorkflowFunction): 
    description:str = Field('StringToBase64Encoder')

    class Arguments(BaseModel):
        plain: str = Field(description="Plain text string to encode")
    
    class Returness(BaseModel):
        encoded: str = Field(default="", description="Base64 encoded string")
    
    args: Optional[Arguments] = None
    rets: Returness = Returness()
    
    def __call__(self,plain:str=''):
        self.args = self.Arguments(plain=plain)
        string_bytes = self.args.plain.encode('utf-8')
        base64_bytes = base64.b64encode(string_bytes)
        self.rets.encoded = base64_bytes.decode('utf-8')
        return self.rets.encoded
    
class Base64ToStringDecoder(Model4LLMs.MermaidWorkflowFunction): 
    description:str = Field('Base64ToStringDecoder')

    class Arguments(BaseModel):
        encoded: str = Field(description="Base64 string to decode")
    
    class Returness(BaseModel):
        plain: str = Field(default="", description="Decoded plain text string")
    
    args: Optional[Arguments] = None
    rets: Returness = Returness()    
        
    def __call__(self,encoded:str=''):
        self.args = self.Arguments(encoded=encoded)
        base64_bytes = self.args.encoded.encode('utf-8')
        string_bytes = base64.b64decode(base64_bytes)
        self.rets.plain = string_bytes.decode('utf-8')
        return self.rets.plain
    
class StringTemplate(Model4LLMs.MermaidWorkflowFunction):
    description: str = Field(default='Dynamic templated string callable')
    
    class Parameter(BaseModel):
        string:str = Field(description='string of f"..."')
        
    class Returness(BaseModel):
        data: str = ''

    para: Parameter
    rets: Returness = Returness()
    
    def build(self):
        self._create_dynamic_call()
        return self

    def _extract_placeholders(self) -> list[str]:
        string = self.para.string.replace(r'{{','')
        string = string.replace(r'}}','')
        return re.findall(r'{(.*?)}', string)

    def _create_dynamic_call(self):
        fields = self._extract_placeholders()
        arg_str = ', '.join([f'{f}:str' for f in fields])
        format_args = ', '.join([f'{f}={f}' for f in fields])

        func_code = f"""
def __call__(self, {arg_str}):
    self.rets.data = self.para.string.format({format_args})
    return self.rets.data
"""

        local_vars = {}
        exec(func_code, {}, local_vars)
        dynamic_call = local_vars['__call__']
        # Assign to the class so the object becomes callable
        dynamic_cls = create_model(f'StringTemplateDynamic{id(self)}', __base__=StringTemplate)
        self.__class__ = dynamic_cls
        setattr(self.__class__, '__call__', dynamic_call)

class StringChunkerByTokenizer(Model4LLMs.MermaidWorkflowFunction):
    description: str = Field('StringChunkerByTokenizer')

    class Arguments(BaseModel):
        text: str = Field(description="Input text to split into token chunks")
        tokenizer: Optional[str] = Field(
            default="cl100k_base",
            description=(
                "Tokenizer/encoding. For tiktoken: 'cl100k_base', 'o200k_base', 'p50k_base', "
                "or 'auto:<openai-model>' (e.g. 'auto:gpt-4o'). For HF: model id like 'gpt2'."
            )
        )
        # Overlaps are percentages of the *target* region (floats).
        left_overlap: float = Field(default=0.05, description="Left overlap as a fraction/percent of target")
        right_overlap: float = Field(default=0.05, description="Right overlap as a fraction/percent of target")
        # Hard cap: full chunk length (left + target + right) must not exceed this
        max_chuck_token: int = Field(default=8192, description="Max tokens per emitted chunk, including overlaps")

        trim_whitespace: bool = Field(default=True, description="Strip whitespace around chunk text")

    class Returness(BaseModel):
        chunks: List[str] = Field(default_factory=list, description="Chunk texts")
        token_counts: List[int] = Field(default_factory=list, description="Token count per chunk (including overlaps)")
        total_tokens: int = Field(default=0, description="Total tokens in input per chosen tokenizer")
        tokenizer_used: str = Field(default="", description="Resolved tokenizer/encoding actually used")
        effective_target_tokens: int = Field(default=0, description="Target (non-overlap) token size actually used")

    args: Optional[Arguments] = None
    rets: Returness = Returness()

    # --- tokenizer loader ---
    def _get_tokenizer(
        self, spec: Optional[str]
    ) -> Tuple[Callable[[str], List[int]], Callable[[List[int]], str], str]:
        # 1) tiktoken (explicit encoding or auto:model)
        if tiktoken is not None:
            if spec and spec.startswith("auto:"):
                model = spec.split("auto:", 1)[1].strip()
                try:
                    enc = tiktoken.encoding_for_model(model)
                    return enc.encode, enc.decode, f"tiktoken:auto:{model}"
                except Exception:
                    pass
            if spec:
                try:
                    enc = tiktoken.get_encoding(spec)
                    return enc.encode, enc.decode, f"tiktoken:{spec}"
                except Exception:
                    pass

        # 2) Hugging Face
        if AutoTokenizer is not None and spec and not spec.startswith("auto:"):
            try:
                hf_tok = AutoTokenizer.from_pretrained(spec, use_fast=True)
                def _encode(s: str) -> List[int]:
                    return hf_tok.encode(s, add_special_tokens=False)
                def _decode(ids: List[int]) -> str:
                    return hf_tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                return _encode, _decode, f"hf:{spec}"
            except Exception:
                pass

        # 3) Whitespace fallback
        table: List[str] = []
        def _ws_encode(s: str) -> List[int]:
            table.clear()
            table.extend(s.split())
            return list(range(len(table)))
        def _ws_decode(ids: List[int]) -> str:
            return " ".join(table[i] for i in ids if 0 <= i < len(table))
        return _ws_encode, _ws_decode, "whitespace:fallback"

    def _normalize_pct(self, x: float) -> float:
        """
        Accepts 0..1, or 0..100 (percent). Values >1 are treated as percent and divided by 100.
        Clamped to [0, 1].
        """
        if x is None:
            return 0.0
        pct = x / 100.0 if x > 1.0 else x
        return max(0.0, min(1.0, pct))

    def __call__(
        self,
        text: str = '',
        tokenizer: str = 'cl100k_base',
        left_overlap: float = 0.05,
        right_overlap: float = 0.05,
        max_chuck_token: int = 8192,
        trim_whitespace: bool = True,
    ):
        # Capture args
        self.args = self.Arguments(
            text=text,
            tokenizer=tokenizer,
            left_overlap=left_overlap,
            right_overlap=right_overlap,
            max_chuck_token=max_chuck_token,
            trim_whitespace=trim_whitespace,
        )

        # Validate
        if self.args.max_chuck_token < 1:
            raise ValueError("max_chuck_token must be >= 1")
        if not (self.args.left_overlap*self.args.max_chuck_token>1 and self.args.right_overlap*self.args.max_chuck_token>1):
            raise ValueError("overlap must be >= 1 token")

        L = self._normalize_pct(self.args.left_overlap)
        R = self._normalize_pct(self.args.right_overlap)

        encode, decode, used_name = self._get_tokenizer(self.args.tokenizer)
        token_ids = encode(self.args.text)
        N = len(token_ids)

        # Compute target (non-overlap) size so that (L + 1 + R) * target <= max_chuck_token
        denom = (1.0 + L + R)
        if denom <= 0:
            denom = 1.0
        base_target = max(1, int(math.floor(self.args.max_chuck_token / denom)))

        chunks: List[str] = []
        counts: List[int] = []

        target_start = 0
        while target_start < N:
            # This slice's target length (last slice may be shorter)
            target_len = min(base_target, N - target_start)

            # Overlaps derived from this slice's *target_len*
            left_tokens  = int(math.floor(L * target_len))
            right_tokens = int(math.floor(R * target_len))

            # Ensure we never exceed max_chuck_token due to rounding
            total_len = left_tokens + target_len + right_tokens
            if total_len > self.args.max_chuck_token:
                overflow = total_len - self.args.max_chuck_token
                # reduce right first, then left
                reduce_r = min(right_tokens, overflow)
                right_tokens -= reduce_r
                overflow -= reduce_r
                if overflow > 0:
                    left_tokens = max(0, left_tokens - overflow)
                total_len = left_tokens + target_len + right_tokens  # recompute

            # Compute slice boundaries (clamp overlaps at text boundaries)
            start = max(0, target_start - left_tokens)
            end   = min(N, target_start + target_len + right_tokens)

            piece_ids = token_ids[start:end]
            piece_text = decode(piece_ids)
            if self.args.trim_whitespace:
                piece_text = piece_text.strip()

            chunks.append(piece_text)
            counts.append(len(piece_ids))

            # advance by the *target* size (non-overlap part)
            target_start += target_len

        self.rets = self.Returness(
            chunks=chunks,
            token_counts=counts,
            total_tokens=N,
            tokenizer_used=used_name,
            effective_target_tokens=base_target,
        )
        # Return primary payload to match your other tools
        return self.rets
  
# @descriptions('Classification Template for performing conditional checks on a target',
#               target='The object or value to be classified',
#               condition_funcs='List of condition functions to evaluate the target')
# class ClassificationTemplate(Model4LLMs.Function):
#     def __call__(self, target: Any, condition_funcs: List[Callable[[Any], Any]] = []) -> List[bool]:
#         # Initialize an empty result list
#         results = []
#         for idx, func in enumerate(condition_funcs):
#             try:
#                 # Attempt to apply the function to the target and ensure it's a boolean result
#                 result = bool(func(target))
#                 results.append(result)
#             except Exception as e:
#                 # Log the error gracefully and append False as a default failure value
#                 print(f"Error in condition function {idx} for target {target}: {e}")
#                 results.append(False)

#         return results

LLMsStore().add_new_class(RegxExtractor)
LLMsStore().add_new_class(StringTemplate)
LLMsStore().add_new_class(StringChunkerByTokenizer)
# LLMsStore().add_new_obj(ClassificationTemplate()).controller.delete()

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
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
        Returness the next chunk of lines.
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

def traverse_files(folder_path, ext='.md', recursive=True):
    ext_files:list[str] = []
    ext_files:list[str] = []

    if recursive:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(ext):
                    ext_files.append(os.path.join(root, file))
                    ext_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path) and file.lower().endswith(ext):
                ext_files.append(full_path)

    return ext_files


def create_file_forcefully(file_path, content=''):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def file2file(in_path,out_path,transfunc=lambda x:x):
    try:
        with open(in_path, 'r', encoding='utf-8') as f: s=f.read()
        create_file_forcefully(out_path,transfunc(s))
        return [in_path,out_path,True]
    except Exception as e:
        print(e)
        return [in_path,out_path,False,e]

def files2files(in_dir,out_dir,ext='.md',transfunc=lambda x:x,overwrite=False):
    for p in traverse_files(in_dir,ext,True):
        op = p.replace(in_dir,out_dir)
        if os.path.exists(op) and not overwrite:continue
        yield file2file(p,op,transfunc)

def files2file(in_dir,out_path,ext='.md',transfunc=lambda x:x,overwrite=False):
    s = ''
    for p in traverse_files(in_dir,ext,True):
        with open(p, 'r', encoding='utf-8') as f: s+=f.read()
    if os.path.exists(out_path) and not overwrite:return False
    create_file_forcefully(out_path,transfunc(s))
    return True
    
