"""File handling utilities for CAB evaluation."""

import json
from typing import Union, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def load_json_or_jsonl_file(file_path: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Load JSON or JSONL file and return its contents.
    
    For JSONL files, returns a list of dictionaries.
    For JSON files, returns a single dictionary.
    
    Args:
        file_path: Path to the file to load
        
    Returns:
        File contents as dict (JSON) or list of dicts (JSONL)
        
    Raises:
        Exception: If file cannot be loaded or parsed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Check if it's a JSONL file (JSON Lines format)
            if file_path.endswith('.jsonl'):
                # Load each line as a separate JSON object
                data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            json_obj = json.loads(line)
                            data.append(json_obj)
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing JSON on line {line_num} in file {file_path}: {e}")
                            logger.error(f"Line content: {line[:100]}...")  # Show first 100 chars for debugging
                            raise
                return data
            else:
                # Standard JSON file - try loading as single JSON object
                return json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error loading issue data from {file_path}: Invalid JSON format - {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading issue data from {file_path}: {e}")
        raise
