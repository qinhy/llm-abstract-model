import os
import ast

def collect_imports_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        node = ast.parse(file.read(), filename=filepath)
    imports = []

    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                imports.append(f"import {alias.name}")
        elif isinstance(stmt, ast.ImportFrom):
            module = stmt.module or ""
            for alias in stmt.names:
                if stmt.level == 0:
                    imports.append(f"from {module} import {alias.name}")
                else:
                    # Handle relative imports with leading dots
                    imports.append(f"from {'.' * stmt.level}{module} import {alias.name}")

    return imports

def collect_all_imports(root_dir):
    all_imports = set()
    for subdir, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.py'):
                full_path = os.path.join(subdir, filename)
                try:
                    imports = collect_imports_from_file(full_path)
                    all_imports.update(imports)
                except Exception as e:
                    print(f"Error processing {full_path}: {e}")
    return sorted(all_imports)

if __name__ == "__main__":
    library_path = './LLMAbstractModel'
    imports = collect_all_imports(library_path)
    imports = [i for i in imports if ' .' not in i]
    print("Collected Imports:")
    for imp in imports:
        print(imp)
