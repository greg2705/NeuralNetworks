import os
import shlex
import subprocess
from pathlib import Path
from typing import List, Tuple

SANDBOX_ROOT = Path("/project/agent_sandbox").resolve()

ALLOWED_COMMANDS = {
    "ls",
    "pwd",
    "cat",
    "head",
    "tail",
    "grep",
    "wc",
    "sed",
    "awk",
    "mkdir",
    "touch",
}

# characters we never allow in commands
FORBIDDEN_TOKENS = {"|", ";", "&&", "||", "`", "$(", ")", ">" , ">>", "<"}

class ShellPolicyError(Exception):
    pass


def _ensure_under_sandbox(path: Path) -> Path:
    path = (SANDBOX_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if SANDBOX_ROOT not in path.parents and path != SANDBOX_ROOT:
        raise ShellPolicyError(f"Path {path} escapes sandbox {SANDBOX_ROOT}")
    return path


def _validate_tokens(tokens: List[str]) -> None:
    # quick check for forbidden shell metacharacters
    for t in tokens:
        for bad in FORBIDDEN_TOKENS:
            if bad in t:
                raise ShellPolicyError(f"Forbidden token '{bad}' in argument '{t}'")


def _normalize_args(cmd: str, args: List[str]) -> Tuple[str, List[str]]:
    if cmd not in ALLOWED_COMMANDS:
        raise ShellPolicyError(f"Command '{cmd}' not allowed")

    _validate_tokens([cmd] + args)

    normalized_args: List[str] = []
    for a in args:
        # treat anything that looks like an option as-is
        if a.startswith("-"):
            normalized_args.append(a)
            continue
        # treat the rest as paths and constrain them
        p = _ensure_under_sandbox(Path(a))
        normalized_args.append(str(p))

    return cmd, normalized_args


def run_restricted(cmd: str, args: List[str], timeout: int = 10) -> subprocess.CompletedProcess:
    """
    Run a single restricted command inside SANDBOX_ROOT.

    Raises ShellPolicyError on policy violations.
    Raises subprocess.CalledProcessError if the command fails (check=True).
    """
    # if someone gives a raw command line, split it first (discourage this in your tool API)
    if args is None:
        tokens = shlex.split(cmd)
        cmd, args = tokens[0], tokens[1:]

    cmd, safe_args = _normalize_args(cmd, args)

    return subprocess.run(
        [cmd, *safe_args],
        cwd=str(SANDBOX_ROOT),
        shell=False,          # critical: no shell interpretation
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
