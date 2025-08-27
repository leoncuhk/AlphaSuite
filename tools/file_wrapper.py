"""
A collection of utility functions for file and data handling.

This module provides tools for:
- Normalizing file paths and generating consistent filenames.
- Serializing complex Python objects (including numpy and pandas types) into a JSON-compatible format.
- Reading and writing files, with automatic handling of JSON and plain text.
- A robust retry decorator (`retry_with_exponential_backoff`) for handling transient API errors.
- Extracting JSON blocks from unstructured text, which is useful for parsing LLM responses.
"""
import logging
import re
import time
from typing import Any, Callable, List, Dict, Union, Optional
from datetime import datetime
import json
import os, traceback
import numpy as np
import pandas as pd
from google.api_core import exceptions as google_exceptions # For specific Google API error handling
from langchain_core.messages import AIMessage

from load_cfg import WORKING_DIRECTORY

def normalize_path(file_path: str) -> str:
    """
    Normalize file path for cross-platform compatibility.
    
    Args:
    file_path (str): The file path to normalize
    
    Returns:
    str: Normalized file path
    """
    if WORKING_DIRECTORY not in file_path:
        file_path = os.path.join(WORKING_DIRECTORY, file_path)
    return os.path.normpath(file_path)

def generate_filename(ticker: str, tool_name: str, extension:str="json") -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join(WORKING_DIRECTORY, f"{ticker}_{tool_name}_{timestamp}.{extension}")
    return filename

def convert_to_json_serializable(data):
    """Recursively converts Timestamps and NaNs to JSON-serializable formats."""
    if isinstance(data, dict):
        return {convert_to_json_serializable(key): convert_to_json_serializable(value)
                for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_json_serializable(v) for v in data]
    elif isinstance(data, pd.Timestamp):  # Handle Timestamps
        return data.strftime('%Y-%m-%d %H:%M:%S')
    elif pd.isna(data):  # Check for NaN values
        return None       # or another suitable JSON representation like "NaN"
    elif isinstance(data, np.int64): # Handle np.int64 which is not JSON serializable
        return int(data)
    elif isinstance(data, np.float64): # Handle np.int64 which is not JSON serializable
        return float(data)
    elif isinstance(data, (np.ndarray, np.generic)): # Handle numpy data
        return data.tolist()
    elif hasattr(data, "to_dict"): # Handle objects with to_dict() function, for example, yf.Ticker
        return data.to_dict()

    try:
        json.dumps(data)  # Check if object is directly serializable
        return data
    except (TypeError, OverflowError):
        return str(data) # convert all other unserializable objects to strings


class FileEdit:
    def write_document(self, content: str, ticker: str, tool_name: str, extension: str = "md") -> str:  # Takes ticker, now includes extension
        filename = generate_filename(ticker, tool_name, extension)
        try:
            # Ensure the directory for the file exists before writing.
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                if isinstance(content, str):
                    f.write(content)
                else:
                    json.dump(content, f, indent=4)  # Handles lists or dictionaries
            return filename
        except Exception as e:
            return f"Error saving file: {e}"

class DocumentLoader:
    def read_document(self, file_name: str, start: Optional[int] = None, end: Optional[int] = None) -> Union[str, List[Dict[str, Any]]]:
        """Reads a document and returns its contents or a section of its contents. It can handle various content types and provides error messages for invalid file paths or formats. 

        Args:
            file_name: The name of the file to read (in WORKING_DIRECTORY().
            start (optional): the starting line number (inclusive) to read from.
            end (optional): The ending line number (exclusive) to read to.

        Returns:
            A dictionary or lines of text containing the content of the document or a section of it, or an error message if there's a problem.
        """
        file_path = normalize_path(file_name)
        if not os.path.exists(file_path):
            return {"error": f"File '{file_name}' not found in '{WORKING_DIRECTORY}'"}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    content = json.load(f)  # Attempt to parse JSON
                    if isinstance(content, list) and all(isinstance(item, dict) for item in content):  # Check if it's a List[Dict]
                        return content[start:end] if isinstance(content, list) else content # Return the slice if requested
                    elif isinstance(content, dict):  # Regular JSON object
                        return content
                    else:
                        return {"error": "Not a JSON array or object: {content}"}
                except json.JSONDecodeError:
                    f.seek(0) # Rewind to beginning for text reading
                    content = f.read()
                    if start is not None or end is not None:
                        lines = content.splitlines()
                        content = "\n".join(lines[start:end])
                    return content

        except Exception as e:
            return {"error": f"Error reading file: {e}"}


def retry_with_exponential_backoff(
    initial_delay: float = 2.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    max_retries: int = 5,
    errors: tuple = (Exception,),
) -> Any:
    """A decorator to retry a function with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            num_retries = 0
            current_delay = initial_delay

            while True:
                try:
                    return func(*args, **kwargs)
                except errors as e:
                    if num_retries >= max_retries:
                        raise Exception(f"Max retries exceeded for {func.__name__} after {num_retries} attempts with error: {e}")

                    num_retries += 1
                    sleep_duration = current_delay
                    handling_message = ""

                    # --- Specific handling for Google API exceptions ---
                    if isinstance(e, google_exceptions.ResourceExhausted):
                        # Try to parse a specific retry delay from the error message
                        match = re.search(r"retry_delay\s*{\s*seconds:\s*(\d+)\s*}", str(e))
                        if match:
                            api_suggested_seconds = int(match.group(1))
                            sleep_duration = api_suggested_seconds + 3.0  # Add a buffer
                            handling_message = f"Using API suggested delay: {api_suggested_seconds}s + 3s buffer."
                        elif jitter:
                            sleep_duration += (num_retries * 0.5)

                    elif isinstance(e, google_exceptions.InternalServerError):
                        handling_message = "Handling InternalServerError (500)."
                        # For 500 errors, be more patient on the first retry.
                        if num_retries == 1:
                            sleep_duration = max(current_delay, 5.0)
                        if jitter:
                            sleep_duration += (num_retries * 0.5)

                    # --- Standard jitter for all other specified errors ---
                    elif jitter:
                        sleep_duration += (num_retries * 0.5)

                    error_type = type(e).__name__
                    error_msg = str(e)[:150]
                    print(
                        f"Retrying {func.__name__} in {sleep_duration:.2f}s (attempt {num_retries}/{max_retries}) "
                        f"due to {error_type}: {error_msg}... {handling_message}"
                    )
                    time.sleep(sleep_duration)
                    current_delay *= exponential_base

        return wrapper
    return decorator


def extract_json_blocks(text: str) -> List[str]:
    """
    Extracts JSON blocks from a text string, handling multiple blocks and surrounding text.

    Args:
        text: The input text string.

    Returns:
        A list of JSON strings, or an empty list if no valid JSON blocks are found.
    """
    if isinstance(text, list):
        return [json.dumps(item) for item in text]  # Convert each item in the list to a JSON string

    cleaned_text = remove_json_marker(text)

    # Attempt 1: Try to parse the entire cleaned text as a single JSON entity (object or array)
    try:
        # Validate if the entire cleaned_text is a single valid JSON
        json.loads(cleaned_text)
        # If successful, return it as the single block
        if isinstance(cleaned_text, list):
            return cleaned_text
        else:
            return [cleaned_text]
    except json.JSONDecodeError:
        # If the entire text is not a single JSON, proceed to find embedded blocks
        print(f"extract_json_blocks: Entire text is not a single JSON. Searching for embedded blocks.")
        pass # Fall through to the loop-based extraction

    # no valid json, search for json blocks
    json_strings = []
    i = 0
    while i < len(text):
        char = text[i]
        if char in ['{', '[']:
            json_str = extract_balanced_json_block(text, i, char, '}' if char == '{' else ']') # Use extract_balanced_json_block
            if json_str:
                json_strings.append(json_str)
                i += len(json_str) - 1
            else:
                i += 1
        else:
            i += 1

    #print(f"{json_strings=}")
    json_blocks = [] # Initialize json_blocks here
    for json_str in json_strings:
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                for item in data:
                    json_blocks.append(json.dumps(item))
            else:
                json_blocks.append(json_str)
        except json.JSONDecodeError:
            print(f"Error decoding JSON in extract_json_blocks.")  # original text:\n{json_str}")

    if json_blocks:
        return json_blocks

    print(f"extract_json_blocks: No valid JSON blocks found using balancing. Original text (first 100 chars): {text[:100]}")
    return [] # Return empty list if no valid JSON found

def extract_balanced_json_block(text: str, start_index: int, open_char: str, close_char: str) -> Optional[str]:
    """
    Helper to extract a balanced JSON block (object or array) starting at start_index.

    Args:
        text (str): The string potentially containing a JSON object or array.
        start_index (int): The starting index in the text to look for the open_char.
        open_char (str): The opening character of the JSON block (e.g., '{' or '[').
        close_char (str): The closing character of the JSON block (e.g., '}' or ']').

    Returns:
        Optional[str]: The extracted JSON block as a string, or None if not found or unbalanced.
    """
    balance = 0
    in_string = False
    string_delimiter = None
    escaped = False
    
    if start_index >= len(text) or text[start_index] != open_char:
        return None # Ensure we start with the expected open_char

    # This loop iterates through the string to find the matching closing character,
    # correctly handling nested structures and quoted strings.
    for i in range(start_index, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == string_delimiter:
                in_string = False
        else:
            if char == '"' or char == "'": # Consider single quotes for robustness, though not standard JSON
                in_string = True
                string_delimiter = char
            elif char == open_char:
                balance += 1
            elif char == close_char:
                balance -= 1
        
        if balance == 0 and not in_string and i >= start_index: # Ensure we've processed at least the open_char
            return text[start_index : i+1]
    return None # Unbalanced or end of string reached before balancing

def remove_json_marker(json_string: str):
    # Remove ```json and ```
    if json_string.startswith("```json\n"):
        json_string = json_string[7:]
    elif json_string.startswith("```\n"):
        json_string = json_string[3:]
    if json_string.endswith("\n```"):
        json_string = json_string[:-4]

    return json_string.strip()

class LLMClient:
    """Handles interactions with the Large Language Model (LLM)."""

    def __init__(self, llm):
        self.llm = llm
        self.imagen_client = None # Initialize Imagen client later if needed

    @retry_with_exponential_backoff(errors=(Exception,))
    def _invoke_with_retry(self, prompt: str):
        """Internal method to handle the LLM invocation with retry."""
        response = self.llm.invoke(prompt)
        if isinstance(response, AIMessage):
            response = response.content
        return response
 
    def get_response(self, prompt: str) -> str:
        """Sends a prompt to the LLM and returns the response content."""
        try:
            return self._invoke_with_retry(prompt)  # Call the method with retry
        except Exception as e:
            return f"An error occurred: {e}"
 
    def get_json_response(self, prompt: str, expected_type: type = dict) -> Optional[Union[Dict, List]]:
        """
        Gets a response from the LLM and robustly parses it as JSON.

        Args:
            prompt: The prompt to send to the LLM.
            expected_type: The expected Python type of the top-level JSON structure (dict or list).

        Returns:
            The parsed JSON as a dictionary or list, or None if parsing fails.
        """
        response_content = self.get_response(prompt)
        json_str = remove_json_marker(response_content)
        if not json_str:
            logging.warning(f"No JSON object found in LLM response. Raw response: {response_content[:300]}")
            return None

        try:
            # Attempt to fix common JSON errors like trailing commas
            json_str_cleaned = json_str.strip()
            if json_str_cleaned.endswith(',}'):
                json_str_cleaned = json_str_cleaned[:-2] + '}'
            elif json_str_cleaned.endswith(',]'):
                json_str_cleaned = json_str_cleaned[:-2] + ']'
            
            parsed_json = json.loads(json_str_cleaned)

            if not isinstance(parsed_json, expected_type):
                logging.warning(f"Parsed JSON is not of expected type {expected_type}. Got {type(parsed_json)}.")
                # Handle case where a single dict is returned instead of a list of one dict
                if expected_type is list and isinstance(parsed_json, dict):
                    logging.info("Attempting to wrap single dictionary in a list.")
                    return [parsed_json]
                return None
            
            return parsed_json
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON response from LLM: {e}\nOriginal Response: {response_content[:500]}\nCleaned String: {json_str[:500]}")
            return None
