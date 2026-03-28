# import ast

# with open('src/build_features.py') as f:
#     src = f.read()

# tree = ast.parse(src)

# print("=== Functions ===")
# for node in ast.walk(tree):
#     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
#         print(f"  {node.name} (line {node.lineno})")

# print()
# print("=== Imports ===")
# for node in ast.walk(tree):
#     if isinstance(node, ast.Import):
#         for n in node.names:
#             print(f"  import {n.name}")
#     elif isinstance(node, ast.ImportFrom):
#         names = [n.name for n in node.names]
#         print(f"  from {node.module} import {', '.join(names)}")

###

# import ast

# with open('src/build_features.py') as f:
#     src = f.read()

# # Find extract_p5_row function and print its source
# lines = src.split('\n')
# in_func = False
# for i, line in enumerate(lines):
#     if 'def extract_p5_row' in line:
#         in_func = True
#     if in_func:
#         print(f"{i+1:4}: {line}")
#     if in_func and i > 82 and line.startswith('def '):
#         break

###

# import ast

# with open('src/build_features.py') as f:
#     lines = f.readlines()

# in_func = False
# for i, line in enumerate(lines):
#     if 'def load_and_resample' in line:
#         in_func = True
#     if in_func:
#         print(f"{i+1:4}: {line}", end='')
#     if in_func and i > 45 and line.startswith('def '):
#         break

###

# import pandas as pd

# df = pd.read_csv('data/processed/features_fixed.csv')
# expected = [(g, c) for g in range(1, 10) for c in range(1, 7)]
# actual = set(zip(df['group'], df['case']))
# missing = [(g, c) for g, c in expected if (g, c) not in actual]
# print("Missing:", missing)


###

# import ast

# with open('src/models/train.py') as f:
#     src = f.read()

# tree = ast.parse(src)

# print("=== Functions ===")
# for node in ast.walk(tree):
#     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
#         print(f"  {node.name} (line {node.lineno})")

# print()
# print("=== Imports ===")
# for node in ast.walk(tree):
#     if isinstance(node, ast.Import):
#         for n in node.names:
#             print(f"  import {n.name}")
#     elif isinstance(node, ast.ImportFrom):
#         names = [n.name for n in node.names]
#         print(f"  from {node.module} import {', '.join(names)}")

###

# import os

# base = r'C:\Users\sharu\ML Projects\IMES Project\mlruns\857627361193563341\198c7f83c7934c9188af27f50a1e8bbe\artifacts'
# for root, dirs, files in os.walk(base):
#     level = root.replace(base, '').count(os.sep)
#     indent = '  ' * level
#     print(f'{indent}{os.path.basename(root)}/')
#     for f in files:
#         print(f'{indent}  {f}')

###

from pathlib import Path

models_dir = Path('data/processed/models')
for f in sorted(models_dir.glob('*.joblib')):
    size_kb = f.stat().st_size / 1024
    print(f'{f.name:<60} {size_kb:>8.1f} KB')