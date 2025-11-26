
#!/usr/bin/env python3

import sys
from pathlib import Path

# プロジェクトのルートディレクトリをシステムパスに追加
sys.path.append(str(Path(__file__).parent))

from src.main import main

if __name__ == "__main__":
    main()
