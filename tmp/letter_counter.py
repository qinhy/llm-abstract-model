from fastmcp import FastMCP

mcp = FastMCP("Letter Counter")

@mcp.tool()
def letter_counter(word: str, letter: str) -> int:
    return word.lower().count(letter.lower())

if __name__ == "__main__":
    mcp.run(transport="stdio")