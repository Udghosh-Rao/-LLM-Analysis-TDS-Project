import os
import subprocess

from langchain_core.tools import tool

OUTPUT_LIMIT = 10_000


@tool
def run_code(code: str) -> dict:
    """
    Execute Python code in a sandboxed subprocess and return its output.

    Parameters
    ----------
    code : str
        Valid Python source code to execute.

    Returns
    -------
    dict
        {
            "stdout":      "<program output>",
            "stderr":      "<errors if any>",
            "return_code": 0
        }
    """
    try:
        work_dir = os.path.abspath("LLMFiles")
        os.makedirs(work_dir, exist_ok=True)
        filepath = os.path.join(work_dir, "runner.py")

        with open(filepath, "w") as f:
            f.write(code)

        proc = subprocess.Popen(
            ["uv", "run", filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=work_dir,
        )
        stdout, stderr = proc.communicate(timeout=60)

        if len(stdout) > OUTPUT_LIMIT:
            stdout = stdout[:OUTPUT_LIMIT] + "\n... [TRUNCATED]"
        if len(stderr) > OUTPUT_LIMIT:
            stderr = stderr[:OUTPUT_LIMIT] + "\n... [TRUNCATED]"

        return {"stdout": stdout, "stderr": stderr, "return_code": proc.returncode}
    except subprocess.TimeoutExpired:
        proc.kill()
        return {"stdout": "", "stderr": "Execution timed out after 60 seconds.", "return_code": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "return_code": -1}