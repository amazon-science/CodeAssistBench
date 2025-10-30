#!/usr/bin/env python3
"""
Script to add task_id to each issue in the CAB evaluation datasets.
"""

import json
import os


def add_task_ids_to_dataset(input_file, output_file, task_prefix):
    """
    Add task_id to each JSON object in a JSONL file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file  
        task_prefix: Prefix for task_id (e.g., "cab_lenient" or "cab_strict")
    """
    print(f"Processing {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist!")
        return False
    
    updated_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Found {len(lines)} issues to process")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                # Parse JSON object
                issue = json.loads(line)
                
                # Create new ordered dictionary with task_id first
                updated_issue = {"task_id": f"{task_prefix}_{i + 1}"}
                
                # Add all other keys from original issue
                for key, value in issue.items():
                    updated_issue[key] = value
                
                # Convert back to JSON string
                updated_lines.append(json.dumps(updated_issue, ensure_ascii=False))
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {i + 1}: {e}")
                return False
        
        # Write updated content to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in updated_lines:
                f.write(line + '\n')
        
        print(f"Successfully updated {len(updated_lines)} issues in {output_file}")
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False


def main():
    """Main function to process both dataset files."""
    datasets = [
        {
            'input': 'dataset/cab_verified_lenient.jsonl',
            'output': 'dataset/cab_verified_lenient.jsonl',
            'prefix': 'cab_lenient'
        },
        {
            'input': 'dataset/cab_verified_strict.jsonl', 
            'output': 'dataset/cab_verified_strict.jsonl',
            'prefix': 'cab_strict'
        }
    ]
    
    all_success = True
    
    for dataset in datasets:
        success = add_task_ids_to_dataset(
            dataset['input'],
            dataset['output'], 
            dataset['prefix']
        )
        
        if not success:
            all_success = False
            print(f"Failed to process {dataset['input']}")
        else:
            print(f"✓ Successfully processed {dataset['input']}")
        print("-" * 50)
    
    if all_success:
        print("✓ All datasets processed successfully!")
    else:
        print("✗ Some datasets failed to process")
        
    return all_success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
