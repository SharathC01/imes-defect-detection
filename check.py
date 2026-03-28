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

import pandas as pd

df = pd.read_csv('data/processed/features_fixed.csv')
expected = [(g, c) for g in range(1, 10) for c in range(1, 7)]
actual = set(zip(df['group'], df['case']))
missing = [(g, c) for g, c in expected if (g, c) not in actual]
print("Missing:", missing)