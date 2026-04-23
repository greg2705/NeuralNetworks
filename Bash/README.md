# Restricted Shell Skill for LLM Agents

This document describes how the agent’s `shell` skill works: what it can do, where it can do it, and which protections are enforced by the backend wrapper rather than by the model.

## How Claude Code do

Claude Code uses a permission system with allow, ask, and deny rules, and those rules are evaluated in deny -> ask -> allow order, with the first match winning. Bash commands normally require approval, while some read-only commands are treated specially and can run without prompts.


## High‑level design

- The agent does **not** have a full shell.
- The agent can only execute a **small set of allowed commands** (allowlist).
- All commands are confined to a single **sandbox directory**.
- `python` is **not** exposed as a generic “run arbitrary code” command.
- Validation and enforcement happen in the **backend shell wrapper**, not in prompts.

This design follows general recommendations for untrusted command execution: use allowlists, validate inputs, and keep execution inside a confined environment.[web:48][web:40]

## Sandbox directory

All operations are restricted to one project subfolder, e.g.:

- `AGENT_ROOT = /project/agent_sandbox`

Rules:

- The shell wrapper runs all commands with `cwd = AGENT_ROOT`.
- Any path argument is normalized (e.g. `realpath`) and **must start with `AGENT_ROOT`**.
- Any attempt to use `..` to escape the directory, or any absolute path outside `AGENT_ROOT`, is rejected.

This is a logical “jail” that prevents the agent from touching other parts of the filesystem.[web:53]

## Allowed operations

The shell skill focuses on safe read/write operations and simple file management, for example:

- **Introspection**
  - `pwd`
  - `ls`, `ls -l` in subdirectories of `AGENT_ROOT`

- **Read‑only file access**
  - `cat <file>`
  - `head <file>`, `tail <file>`
  - `grep <pattern> <file>`
  - `wc <file>`
  - `sed`, `awk` without in‑place `-i` edits

- **Creating folders and files**
  - `mkdir <dir>` (inside `AGENT_ROOT`)
  - `touch <file>` (inside `AGENT_ROOT`)

Any command outside this explicit allowlist is rejected by the wrapper.

### Explicitly disallowed patterns

Even within allowed commands, some patterns are rejected, for example:

- `;`, `&&`, `||`, pipes, backticks, or `$(...)` (no composing multiple commands).
- Redirections like `>`, `>>` to paths outside `AGENT_ROOT`.
- Any use of `rm`, `chmod`, `chown`, etc. (these commands are not on the allowlist at all).

The wrapper parses the command and refuses it if it contains disallowed constructs.[web:48][web:40]

## Why generic `python` is not allowed

If the agent can both:

- **Write files** (e.g. `echo`/`cat`/`touch` into a script), and
- **Run arbitrary `python`**,  

then it can create a script containing dangerous operations (`os.system("rm -rf ...")`, network calls, etc.) and execute it, bypassing your command allowlist.

To avoid this:

- There is **no generic `python` command** in the shell skill.
- Instead, any Python logic you need (tests, checks, scripts) is exposed as **separate, narrowly scoped tools** implemented in your backend, for example:
  - `run_pytest`: runs a fixed command like `python -m pytest tests/audit/*.py` in `AGENT_ROOT`.
  - `run_static_checks`: runs a fixed linter or checker.

The agent can call these tools, but it cannot construct arbitrary Python command lines.

## Shell wrapper responsibilities

The shell skill is implemented by a backend wrapper that:

1. **Parses the request**

   - Receives structured input, e.g.:

     ```json
     { "cmd": "ls", "args": ["-l", "reports/"] }
     ```

   - Rejects raw unstructured strings where parsing would be ambiguous.

2. **Checks the command name**

   - `cmd` must be in a hardcoded `ALLOWED_COMMANDS` list.
   - If not, the wrapper rejects the request.

3. **Validates arguments**

   - Normalizes all paths and ensures they remain under `AGENT_ROOT`.
   - Rejects special characters and operators (`;`, `&&`, `|`, backticks, `$(...)`).
   - Rejects in‑place edits (`sed -i`, etc.).

4. **Executes safely**

   - Uses `subprocess.run([...], cwd=AGENT_ROOT, shell=False, timeout=...)`.
   - Captures stdout/stderr and returns them to the agent.
   - Logs all attempted and executed commands for audit.

This follows known secure‑execution patterns for LLM tools: strict allowlists, argument validation, no `shell=True`, and full logging.[web:48][web:40]

## Example policy table

You can embed this table directly in `skills.md`:

```markdown
### Shell skill policy

| Command / Pattern              | Policy   | Notes                                 |
|--------------------------------|----------|---------------------------------------|
| `pwd`, `ls`, `ls -l`           | allow    | Only under `AGENT_ROOT`               |
| `cat`, `head`, `tail`, `grep`  | allow    | Read‑only, paths under `AGENT_ROOT`   |
| `wc`, `sed`, `awk`             | allow    | No in‑place `-i` edits                |
| `mkdir <dir>`                  | allow    | Directory under `AGENT_ROOT`          |
| `touch <file>`                 | allow    | File under `AGENT_ROOT`               |
| Any other command (incl. `git`)| deny     | Not on allowlist                      |
| Pipes, `;`, `&&`, `||`, `$()`  | deny     | Multi‑command or subshell not allowed |
| Absolute path outside root     | deny     | Must stay in `AGENT_ROOT`             |
| Generic `python` invocation    | deny     | Use dedicated tools instead           |
```

## Summary for the agent spec

- The `shell` skill is **minimal**: read/write + basic file/folder ops inside one directory.
- The agent **cannot**:
  - Run arbitrary shell pipelines.
  - Escape its sandbox directory.
  - Run generic `python` or other interpreters.
- All safety is enforced **in code** in the shell wrapper, not by trusting the LLM.

You can paste and adapt this Markdown into your internal `skills.md` so other engineers know exactly what the shell skill guarantees and what it refuses.
