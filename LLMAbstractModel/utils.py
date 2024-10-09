from multiprocessing import Lock
import random
import re
import uuid
from pydantic import Field
from typing import Any, Dict, Optional, List
import mt5

from .BasicModel import BasicModel
from .LLMsModel import Model4LLMs
descriptions = Model4LLMs.Function.param_descriptions
@descriptions('Extract text by regx pattern',
              text='input text')
class RegxExtractor(Model4LLMs.Function):
    regx:str = Field(description='regx pattern')
    def __call__(self,text:str):
        return self.extract(text)
    
    def extract(self,text)->str:
        matches = re.findall(self.regx, text, re.DOTALL)        
        if not self._try_binary_error(lambda:matches[0]):
            self._log_error(ValueError(f'cannot match {self.regx} at {text}'))
            return text
        return matches[0]

@descriptions('String Template for format function',
              args='string.format args')
class StringTemplate(Model4LLMs.Function):
    string:str = Field(description='string of f"..."')
    def __call__(self,*args):
        return self.string.format(*args)

class MT5Action:
    #     timeout 
    #     retry_times on timeout
    #     retry_times on error
    
    def __init__(self,account:Model4LLMs.MT5Account, retry_times_on_error=3) -> None:       
        # do set_account at first
        self.uuid = uuid.uuid4()
        self._account:Model4LLMs.MT5Account = account
        self.retry_times_on_error = retry_times_on_error

    def set_account(self,account_id,password,account_server):
        # do this at first
        self._account.account_id=account_id
        self._account.password=password
        self._account.account_server=account_server
        return self
    
    def _run(self):
        res = None
        try:
            res = self.run()
            self.on_end(res)
        except Exception as e:
            print(e)
            return self._on_error(e)
        return res

    def run(self):
        print('do your action at here with mt5')
    
    def on_error(self,e):
        pass
        # print('not implement')

    def _on_error(self,e):
        self.retry_times_on_error-=1
        if self.retry_times_on_error>0:
            time.sleep(1)
            return self._run()
        else:
            return self.on_error(e)

    def on_end(self,res):
        pass
        # print('not implement')

class MT5Manager:
    class TerminalLock:
        def __init__(self,exe_path="path/to/your/terminal64.exe"):
            self.exe_path=exe_path
            self._lock = Lock()
        def acquire(self):
            # print("acquired", self)
            self._lock.acquire()
        def release(self):
            # print("released", self)
            self._lock.release()
        def __enter__(self):
            self.acquire()
        def __exit__(self, type, value, traceback):
            self.release()

    def __init__(self) -> None:
        self.results:Dict[str,List[Any]] = {}
        self.terminals:Dict[str,List[MT5Manager.TerminalLock]] = {}
    # {
    #     'TitanFX':[
    #         MT5Manager.TerminalLock(exe_path="path/to/your/terminal64.exe")
    #     ],
    #     'XMTrading':[],
    # }
    def add_terminal(self, account_server='XMTrading', exe_path="path/to/your/terminal64.exe"):
        if account_server not in self.terminals:self.terminals[account_server]=[]
        self.terminals.get(account_server,[]).append(
            MT5Manager.TerminalLock(exe_path=exe_path))
    
    def _get_terminal_lock(self, account_server='XMTrading'):
        broker = account_server.split('-')[0]
        t_locks = self.terminals.get(broker,[])
        if len(t_locks)==0:raise ValueError('the broker is not support!')
        return random.choice(t_locks)

    def do(self, action:MT5Action):
        # m = Manager()
        # m.addExe

        # m.do(
        #     new Action class do some thing
        #     timeout 
        #     retry_times on timeout
        #     retry_times on error
        # )
        
        # get lock
        l = self._get_terminal_lock(action._account.account_server.get_secret_value())
        try:
            l.acquire()
            if not mt5.initialize(path=l.exe_path):
                raise ValueError(f"Failed to initialize MT5 for executable path: {l.exe_path}")
            
            if action._account is None:raise ValueError('_account is not set')
            action._account.is_valid()
            account = action._account
    
            if not mt5.login(account.account_id,
                             password=account.password.get_secret_value(),
                             server=account.account_server.get_secret_value()):                
                raise ValueError(f"Failed to log in with account ID: {account.account_id}")
            
            if action.uuid not in self.results:self.results[action.uuid]=[]
            self.results[action.uuid].append(action._run())            
        finally:
            mt5.shutdown()  # Ensure shutdown is called even if an error occurs
            l.release()
        # release lock

@descriptions('...',args='...')
class MT5CopyLastRates(Model4LLMs.Function):
    # tools.CopyLastRates().set_account(**acc.model_dump())(symbol=c, timeframe=timeframe.value, count=int(bars.value))
    account:Model4LLMs.MT5Account = Field(description='...')
    symbol:str
    timeframe:str
    count:int
    def __call__(self,*args):
        MT5Manager().run(self.get_action())

    def get_action(self):
        return MT5Action(self.account)

@descriptions('...',args='...')
class MT5MakeOder(Model4LLMs.Function):
    account:Model4LLMs.MT5Account = Field(description='...')
    def __call__(self,Symbol:str,EntryPrice:float,TakeProfitPrice:float,ProfitRiskRatio:float=2):
        # make book
        book = Model4LLMs.Book()
        MT5Manager().run(self.get_action(book))

    def get_action(self,book):
        return MT5Action(self.account)
    
# @descriptions('...',args='...')
# class MT5BookSend(Model4LLMs.Function):
#     account:Model4LLMs.MT5Account = Field(description='...')
#     def __call__(self,book:Model4LLMs.Book):
#         book.send()

# @descriptions('...',args='...')
# class MT5BookClose(Model4LLMs.Function):
#     account:Model4LLMs.MT5Account = Field(description='...')
#     def __call__(self,book:Model4LLMs.Book):
#         book.close()  



class TextFile(BasicModel):
    file_path: str
    chunk_lines: int = Field(default=1000, gt=0, description="Number of lines per chunk, must be greater than 0")
    overlap_lines: int = Field(default=100, ge=0, description="Number of overlapping lines between chunks, must be non-negative")
    current_position: int = Field(default=0, ge=0, description="Current position in the file")
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