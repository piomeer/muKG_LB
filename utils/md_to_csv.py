#!/usr/bin/env python3
"""Convert MD files in output/results/ that contain CSV-structured data back to .csv files."""
import os
import re
import csv

ROOT = os.path.join(os.path.dirname(__file__), '..', 'output', 'results')
converted = 0

def is_csv_like_line(line: str) -> bool:
    """Check if a line looks like CSV (comma-separated with numeric values)."""
    parts = line.split(',')
    if len(parts) < 3:
        return False
    return True

def extract_csv_from_md(content: str) -> str:
    """Extract CSV data from markdown content.
    
    Strategy:
    1. If the file starts with a CSV header line (contains commas and looks like header), it's raw CSV saved as .md
    2. If it's a markdown table, convert to CSV
    """
    lines = content.strip().split('\n')
    csv_lines = []
    
    # Case 1: Raw CSV saved as .md (no markdown formatting)
    if len(lines) >= 2 and ',' in lines[0] and '|' not in lines[0] and '#' not in lines[0]:
        # Collect only CSV-formatted lines (skip markdown separators like ---)
        for line in lines:
            stripped = line.strip()
            if stripped and ',' in stripped and not stripped.startswith('---') and not stripped.startswith('#'):
                csv_lines.append(stripped)
    
    # Case 2: Markdown table format
    elif any('|' in l for l in lines[:3]):
        # Skip header separator (|---|...|)
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('|') and not stripped.replace('|','').replace('-','').replace(' ','').strip() == '':
                # Convert markdown row to CSV
                parts = [p.strip() for p in stripped.split('|')]
                # Remove empty first/last from split
                parts = [p for p in parts if p]
                if parts:
                    csv_lines.append(','.join(parts))
    
    return '\n'.join(csv_lines)

def main():
    global converted
    for dirpath, dirnames, filenames in os.walk(ROOT):
        for fname in filenames:
            if not fname.endswith('.md'):
                continue
            md_path = os.path.join(dirpath, fname)
            csv_path = md_path.replace('.md', '.csv')
            
            # Skip if .csv already exists
            if os.path.exists(csv_path):
                continue
            
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            csv_content = extract_csv_from_md(content)
            if not csv_content or len(csv_content.split('\n')) < 2:
                # Not CSV-structured, skip
                continue
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                f.write(csv_content)
            
            converted += 1
            rel_path = os.path.relpath(csv_path, ROOT)
            print(f'[OK] {rel_path} ({len(csv_content.split(chr(10)))} lines)')

    print(f'\nTotal converted: {converted} files')

if __name__ == '__main__':
    main()