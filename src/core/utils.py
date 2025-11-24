import sys
import os

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
        # If running from source, we might be in the root directory or src
        # We assume the 'src' folder is in the current directory or one level down/up
        # But based on the project structure:
        # root/
        #   src/
        #     assets/
        #   main_gui.py
        
        # If we are running main_gui.py from root, base_path is root.
        # relative_path passed should be 'src/assets/welcome.png' if we want to be consistent?
        # Or we can try to find 'src'
        
        if not os.path.exists(os.path.join(base_path, relative_path)):
            # Try looking in src if not found in root
            if os.path.exists(os.path.join(base_path, 'src', relative_path)):
                return os.path.join(base_path, 'src', relative_path)
                
    return os.path.join(base_path, relative_path)
