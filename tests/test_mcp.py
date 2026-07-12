"""
test_mcp.py
Unit and integration verification for the Tdqeq MCP server.
"""
import json
import pytest
from tdqeq.mcp_server import mcp

@pytest.mark.anyio
async def test_mcp_tool_registration():
    """Verify that the extract_tables tool is registered on the FastMCP server."""
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "extract_tables" in tool_names
    
    # Find the tool and verify its input arguments schema
    extract_tool = next(t for t in tools if t.name == "extract_tables")
    assert "pdf_path" in extract_tool.inputSchema["properties"]
    assert "accelerate" in extract_tool.inputSchema["properties"]
    assert "start_page" in extract_tool.inputSchema["properties"]
    assert "end_page" in extract_tool.inputSchema["properties"]

@pytest.mark.anyio
async def test_mcp_tool_invalid_file():
    """Verify that calling the tool with a non-existent file returns an error JSON."""
    result = await mcp.call_tool("extract_tables", arguments={"pdf_path": "non_existent_file.pdf"})
    assert result is not None
    
    # FastMCP call_tool returns a tuple (contents, meta)
    contents, _ = result
    assert len(contents) > 0
    text_content = contents[0].text
    payload = json.loads(text_content)
    assert "error" in payload
    assert "File not found" in payload["error"]
