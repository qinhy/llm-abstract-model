import re
from pydantic import BaseModel, Field
from typing import Optional, List
from .BasicModel import BasicModel

class TextFile(BasicModel):
    file_path: str
    chunk_lines: int = Field(default=1000, gt=0, description="Number of lines per chunk, must be greater than 0")
    overlap_lines: int = Field(default=100, ge=0, description="Number of overlapping lines between chunks, must be non-negative")
    current_position: int = Field(default=0, ge=0, description="Current position in the file")
    line_count: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
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

    def __next__(self) -> List[str]:
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

# Example usage:
# text_file = TextFile(file_path="example.txt")
# for chunk in text_file:
#     print(chunk)
# text_file.close()