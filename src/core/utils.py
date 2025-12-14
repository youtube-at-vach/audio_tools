import sys
import os
import math


_SI_PREFIXES = {
    -24: "y",
    -21: "z",
    -18: "a",
    -15: "f",
    -12: "p",
    -9: "n",
    -6: "µ",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
    18: "E",
    21: "Z",
    24: "Y",
}


def format_si(value, unit: str = "", sig_figs: int = 4, space: str = " ") -> str:
    """Format a numeric value using SI prefixes (engineering notation).

    - Uses significant figures (like format specifier 'g') on the scaled value.
    - Chooses a prefix in steps of 10^3 so the scaled magnitude is in [1, 1000).
    - Returns '-' for NaN/Inf.
    """
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "-"

    if not math.isfinite(x):
        return "-"

    if x == 0.0:
        return f"0{space}{unit}".rstrip()

    ax = abs(x)

    # Compute engineering exponent (multiple of 3).
    exp3 = int(math.floor(math.log10(ax) / 3.0) * 3)
    exp3 = max(min(exp3, 24), -24)

    scale = 10.0 ** exp3
    scaled = x / scale

    # Handle rounding spillover (e.g., 999.95 m -> 1.000 k).
    if abs(scaled) >= 999.5 and exp3 < 24:
        exp3 += 3
        scale *= 1000.0
        scaled = x / scale

    prefix = _SI_PREFIXES.get(exp3, "")
    number = f"{scaled:.{int(sig_figs)}g}"

    # Avoid displaying '-0' which can happen with rounding.
    if number in ("-0", "-0.0", "-0.00"):
        number = number[1:]

    return f"{number}{space}{prefix}{unit}".rstrip()

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
