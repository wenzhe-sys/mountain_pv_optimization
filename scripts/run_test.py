"""Run tests and save results to file."""
import subprocess
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
r = subprocess.run(
    ["python", "-m", "pytest", "tests/test_branch_and_price.py", "--tb=line", "-v"],
    capture_output=True, text=True, timeout=300
)
with open("test_result.txt", "w", encoding="utf-8") as f:
    f.write(r.stdout)
    f.write("\n---STDERR---\n")
    f.write(r.stderr)
    f.write(f"\n---RETURNCODE={r.returncode}---\n")
print(f"Done. RC={r.returncode}")
print(r.stdout[-500:] if len(r.stdout) > 500 else r.stdout)
