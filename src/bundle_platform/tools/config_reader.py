"""
Config file reading tool for sosreport bundles.

Config files (under etc/) are typically small and static — they capture the
system's configuration at the time of the sosreport. This module provides a
single read_config() function that reads them safely with a line cap to prevent
accidentally loading a huge file (e.g. a generated config or a certificate bundle).

This module is also used by agent.py to implement the read_sos_command tool,
which reads from sos_commands/ instead of etc/. The function is agnostic to the
directory — it just reads whatever relative path it's given.
"""

from pathlib import Path

from bundle_platform.shared.caps import cap_lines

# Maximum lines to return from any config file read.
# Most config files are well under 150 lines. This cap prevents a pathologically
# large file (e.g. /etc/hosts with thousands of entries) from flooding the context.
_MAX_CONFIG_LINES = 150


def read_config(bundle_root: Path, file_path: str) -> str:
    """
    Read a file from the bundle and return its contents as numbered lines.

    Why: Config files are the ground truth for how a system was configured when
    the sosreport was collected. The agent uses this to answer questions like
    "what was the multipath polling interval?" or "what DNS servers were configured?"

    Why numbered lines: Consistent with grep_log() and read_section() output — the
    agent can refer to "line 12" across all tool outputs.

    How: Reads the whole file as text, returns up to _MAX_CONFIG_LINES lines with
    1-indexed line numbers prepended. Appends a truncation notice if the file is
    longer than the cap so the agent knows there is more content.

    This function is also called by dispatch_tool() for the read_sos_command tool,
    which constructs a sos_commands/<name> path and delegates here. The function
    is path-agnostic — it works for any file in the bundle.

    Args:
        bundle_root: Root directory of the unpacked bundle.
        file_path:   Relative path to the file within the bundle.

    Returns:
        Numbered file contents, capped at _MAX_CONFIG_LINES.
        Returns a "not found" message if the file doesn't exist.
        Returns an error message if the file can't be read.
    """
    full_path = bundle_root / file_path
    if not full_path.resolve().is_relative_to(bundle_root.resolve()):
        return f"Error: path is outside bundle: {file_path}"
    if not full_path.exists():
        return f"File not found: {file_path}"

    try:
        lines = full_path.read_text(errors="ignore").splitlines()
    except OSError as e:
        # OSError covers permission denied, I/O errors, and other filesystem issues.
        # Return a descriptive message rather than raising — the agent can decide
        # whether to try a different file or report the issue to the user.
        return f"Cannot read {file_path}: {e}"

    numbered = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))
    return cap_lines(numbered, limit=_MAX_CONFIG_LINES)


def read_sos_command(bundle_root: Path, command_name: str, bundle_type: str) -> str:
    """
    Read output from a captured command in the bundle.

    Why: Bundle types store command output in different directories — RHEL uses
    `sos_commands/` (produced by `sos report`) while ESXi uses `commands/`
    (produced by `vm-support`). This function hides that difference so callers
    always pass a bare command name.

    Args:
        bundle_root:  Root directory of the unpacked bundle.
        command_name: Bare filename of the command output (e.g. "uname").
        bundle_type:  "rhel" or "esxi" (case-insensitive).

    Returns:
        Numbered file contents capped at _MAX_CONFIG_LINES, or an error string.
    """
    subdir = "sos_commands" if bundle_type.lower() == "rhel" else "commands"
    return read_config(bundle_root, f"{subdir}/{command_name}")
