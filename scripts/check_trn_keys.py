
import ast
import json
import os
import sys
import glob

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANG_DIR = os.path.join(PROJECT_ROOT, "src", "assets", "lang")
WIDGETS_DIR = os.path.join(PROJECT_ROOT, "src", "gui", "widgets")
MAIN_WINDOW_FILE = os.path.join(PROJECT_ROOT, "src", "gui", "main_window.py")
MAIN_GUI_FILE = os.path.join(PROJECT_ROOT, "main_gui.py")

# Helpers
def get_json_files():
    return glob.glob(os.path.join(LANG_DIR, "*.json"))

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_duplicate_keys(path):
    """
    Scans a JSON file for duplicate keys using simple line matching.
    This finds strict exact duplicates in the file text.
    """
    keys = set()
    duplicates = set()
    import re
    # Regex to find "key": at the start of a line (ignoring whitespace)
    # This assumes standard formatting like "key": "value"
    pattern = re.compile(r'^\s*"((?:[^"\\]|\\.)+)"\s*:')
    
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            match = pattern.search(line)
            if match:
                key = match.group(1)
                # Unescape escaped quotes if necessary (basic handling)
                key = key.replace('\\"', '"') 
                if key in keys:
                    duplicates.add(key)
                keys.add(key)
    return list(duplicates)

class TrVisitor(ast.NodeVisitor):
    def __init__(self):
        self.keys = set()
        
    def visit_Call(self, node):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            
        if func_name == 'tr':
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                self.keys.add(node.args[0].value)
        
        self.generic_visit(node)

def extract_tr_keys(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=filepath)
        visitor = TrVisitor()
        visitor.visit(tree)
        return visitor.keys
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return set()

def main():
    print("=== Translation Check Script ===")
    
    # 1. Load EN JSON (Source of Truth)
    en_path = os.path.join(LANG_DIR, 'en.json')
    if not os.path.exists(en_path):
        print(f"Error: en.json not found at {en_path}")
        sys.exit(1)
        
    en_data = load_json(en_path)
    en_keys = set(en_data.keys())
    print(f"Loaded {len(en_keys)} keys from en.json")

    # 2. Extract keys from Code
    files_to_scan = []
    # Widgets
    files_to_scan.extend(glob.glob(os.path.join(WIDGETS_DIR, "*.py")))
    # Main Window
    if os.path.exists(MAIN_WINDOW_FILE):
        files_to_scan.append(MAIN_WINDOW_FILE)
    # Main GUI
    if os.path.exists(MAIN_GUI_FILE):
        files_to_scan.append(MAIN_GUI_FILE)
        
    code_keys = set()
    for fp in files_to_scan:
        file_keys = extract_tr_keys(fp)
        code_keys.update(file_keys)
        
    print(f"Found {len(code_keys)} unique tr() keys in {len(files_to_scan)} source files.")
    
    # 3. Check: Code Keys exist in en.json
    missing_in_en = []
    for k in code_keys:
        if k not in en_keys:
            missing_in_en.append(k)
            
    # 4. Check: Other JSONs have all keys from en.json
    json_files = get_json_files()
    missing_translations = {} # filename -> list of missing keys
    
    for jf in json_files:
        fname = os.path.basename(jf)
        if fname == 'en.json':
            continue
            
        data = load_json(jf)
        local_keys = set(data.keys())
        diff = en_keys - local_keys
        if diff:
            missing_translations[fname] = list(diff)
            
    # 5. Check Duplicates (Warning only)
    duplicates_map = {}
    for jf in json_files:
        dups = find_duplicate_keys(jf)
        if dups:
            duplicates_map[os.path.basename(jf)] = dups

    # Reporting
    has_error = False
    
    print("\n--- Check 1: Missing keys in en.json (Used in Code) ---")
    if missing_in_en:
        has_error = True
        print(f"FAIL: {len(missing_in_en)} keys used in code but missing in en.json:")
        for k in sorted(missing_in_en):
            print(f"  - \"{k}\"")
    else:
        print("OK")
        
    print("\n--- Check 2: Missing translations in other languages (Compared to en.json) ---")
    if missing_translations:
        has_error = True
        for fname, keys in missing_translations.items():
            print(f"FAIL: {fname} is missing {len(keys)} keys:")
            # Show first 10
            for k in sorted(keys)[:10]:
                print(f"  - \"{k}\"")
            if len(keys) > 10:
                print(f"  ... and {len(keys)-10} more.")
    else:
        print("OK")

    print("\n--- Check 3: Duplicate Keys (Warning) ---")
    if duplicates_map:
        for fname, keys in duplicates_map.items():
            print(f"WARNING: {fname} has duplicate keys:")
            for k in keys:
                print(f"  - \"{k}\"")
    else:
        print("OK")

    print("\n=== Result ===")
    if has_error:
        print("TEST FAILED")
        sys.exit(1)
    else:
        print("TEST PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main()
