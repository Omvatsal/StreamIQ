"""Single-command launcher for both StreamIQ Backend (FastAPI) and Frontend (Vite).

Usage:
    python run.py
    ./start.sh
"""

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
VENV_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"

# ANSI color codes for clean dual-stream terminal logging
CYAN = "\033[96m"
VIOLET = "\033[95m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"


def get_python_executable() -> str:
    """Use the project's .venv Python if available."""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def get_node_executable() -> str:
    """Prefer native Linux Node inside WSL to avoid Windows/WSL binary mismatches."""
    linux_node = Path("/usr/bin/node")
    if os.name == "posix" and linux_node.exists():
        return str(linux_node)
    found = shutil.which("node")
    if found:
        return found
    raise RuntimeError("Node.js executable not found in PATH or /usr/bin/node")


def stream_output(pipe, prefix: str, color: str) -> None:
    """Stream subprocess stdout/stderr line-by-line with a colored prefix."""
    try:
        for line in iter(pipe.readline, ""):
            if line:
                sys.stdout.write(f"{color}{prefix}{RESET} {line}")
                sys.stdout.flush()
    except ValueError:
        pass
    finally:
        pipe.close()


def terminate_process(proc: subprocess.Popen) -> None:
    """Cleanly terminate a subprocess and its entire process group."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Start StreamIQ backend and frontend together.")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=5173)
    args = parser.parse_args()

    python_bin = get_python_executable()
    node_bin = get_node_executable()
    vite_entry = FRONTEND_DIR / "node_modules" / "vite" / "bin" / "vite.js"

    if not vite_entry.exists():
        print(f"Error: Vite entrypoint not found at {vite_entry}. Run npm install in frontend/.")
        return 1

    print(f"\n{BOLD}{CYAN}=== StreamIQ Full-Stack Control Center ==={RESET}")
    print(f"  {GREEN}Frontend (Vite Dev Server):{RESET}  http://localhost:{args.frontend_port}/")
    print(f"  {GREEN}Integrated Production UI:{RESET}   http://localhost:{args.backend_port}/ui/")
    print(f"  {GREEN}Backend API Docs (Swagger):{RESET} http://localhost:{args.backend_port}/docs")
    print(f"  Press {BOLD}Ctrl+C{RESET} to stop both services.\n")

    backend_cmd = [
        python_bin,
        "-m",
        "uvicorn",
        "backend.app.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.backend_port),
        "--reload",
    ]
    frontend_cmd = [
        node_bin,
        str(vite_entry),
        "--host",
        "0.0.0.0",
        "--port",
        str(args.frontend_port),
    ]

    popen_kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
    }
    if os.name == "posix":
        popen_kwargs["preexec_fn"] = os.setsid

    backend_proc = subprocess.Popen(backend_cmd, cwd=str(ROOT_DIR), **popen_kwargs)
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(FRONTEND_DIR), **popen_kwargs)

    t_backend = threading.Thread(
        target=stream_output, args=(backend_proc.stdout, "[BACKEND] ", CYAN), daemon=True
    )
    t_frontend = threading.Thread(
        target=stream_output, args=(frontend_proc.stdout, "[FRONTEND]", VIOLET), daemon=True
    )
    t_backend.start()
    t_frontend.start()

    def handle_signal(signum, frame):
        print(f"\n{BOLD}Shutting down StreamIQ services...{RESET}")
        terminate_process(frontend_proc)
        terminate_process(backend_proc)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while True:
            b_code = backend_proc.poll()
            f_code = frontend_proc.poll()
            if b_code is not None or f_code is not None:
                break
            threading.Event().wait(0.5)
    except KeyboardInterrupt:
        handle_signal(signal.SIGINT, None)
    finally:
        terminate_process(frontend_proc)
        terminate_process(backend_proc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
