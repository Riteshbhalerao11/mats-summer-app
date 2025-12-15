#!/usr/bin/env python3
"""
Create a Parquet dataset from generated conversation JSON files.

This script recursively scans the generated_conversations directory,
extracts conversation data from all JSON files, and creates a Parquet
file for efficient storage and analysis.

Usage:
    python create_dataset.py
"""

import json
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any


def load_conversations_from_json_files(base_dir: Path = Path("./generated_conversations")) -> List[Dict[str, Any]]:
    """
    Recursively load all conversation JSON files from the base directory.
    
    Args:
        base_dir: Base directory containing conversation JSON files
        
    Returns:
        List of dictionaries containing extracted conversation data
    """
    conversations = []
    
    # Find all JSON files recursively
    json_files = list(base_dir.rglob("*.json"))
    
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract only the 4 required fields
            conversation_data = {
                'scenario_type': data.get('scenario_type'),
                'trust_level': data.get('trust_level'),
                'scenario_context': data.get('scenario_context'),
                'conversation': data.get('parsed_conversation', [])  # List of dicts
            }
            
            conversations.append(conversation_data)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    return conversations


def create_parquet_dataset(output_file: str = "conversations_dataset.parquet"):
    """
    Create a Parquet dataset from all conversation JSON files.
    
    Args:
        output_file: Output Parquet file path
    """
    print("="*80)
    print("CREATING CONVERSATION DATASET")
    print("="*80)
    
    # Load all conversations
    conversations = load_conversations_from_json_files()
    
    if not conversations:
        print("No conversations found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(conversations)
    
    print(f"\nDataset Statistics:")
    print(f"Total Conversations: {len(df)}")
    print(f"\nBy Scenario Type:")
    print(df['scenario_type'].value_counts())
    print(f"\nBy Trust Level:")
    print(df['trust_level'].value_counts())
    
    if 'scenario_context' in df.columns:
        print(f"\nBy Scenario Context:")
        print(df['scenario_context'].value_counts())
    
    # Save to Parquet
    df.to_parquet(output_file, engine='pyarrow', compression='snappy')
    
    print(f"\n{'='*80}")
    print(f"Dataset saved to: {output_file}")
    print(f"File size: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*80}")
    
    # Display sample
    print("\nSample conversation structure:")
    if len(df) > 0:
        sample = df.iloc[0]
        print(f"Scenario Type: {sample['scenario_type']}")
        print(f"Trust Level: {sample['trust_level']}")
        print(f"Scenario Context: {sample.get('scenario_context', 'N/A')}")
        print(f"Number of turns: {len(sample['conversation'])}")
        print(f"First turn: {sample['conversation'][0] if sample['conversation'] else 'N/A'}")


if __name__ == '__main__':
    create_parquet_dataset()
