"""File utility tools for agents."""

import os
from typing import Optional

async def read_file(file_path: str) -> str:
    """
    Read the contents of a file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        Contents of the file as a string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"Successfully read file {file_path}:\n\n{content}"
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"

async def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        
    Returns:
        Success or error message
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Write content to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing to file {file_path}: {str(e)}"

async def list_files(directory: str = '.') -> str:
    """
    List files in a directory.
    
    Args:
        directory: Directory to list files from (defaults to current directory)
        
    Returns:
        List of files in the directory
    """
    try:
        files = os.listdir(directory)
        return f"Files in {directory}:\n\n" + "\n".join([
            f"{'[DIR]' if os.path.isdir(os.path.join(directory, f)) else '[FILE]'} {f}" 
            for f in files
        ])
    except Exception as e:
        return f"Error listing files in {directory}: {str(e)}"