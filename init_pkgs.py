from pathlib import Path
import os

for d in ["src", "src/storage", "src/workload", "src/engines", "src/features", "src/models", "src/evaluation"]:
    p = Path("C:/Users/jainy/.gemini/antigravity/scratch/learned-cost-estimation") / d / "__init__.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('"""Package init."""\n', encoding="utf-8")
