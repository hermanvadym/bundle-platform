import tarfile
from pathlib import Path


def detect_bundle_type(archive_path: Path) -> str:
    """
    Detect bundle type (RHEL or ESXi) by peeking at tar archive member names.

    Args:
        archive_path: Path to the tar archive.

    Returns:
        "rhel" if sosreport format detected.
        "esxi" if ESXi vm-support format detected.

    Raises:
        FileNotFoundError: If archive_path does not exist.
        ValueError: If bundle type cannot be determined.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    with tarfile.open(archive_path, "r:*") as tf:
        members = tf.getnames()

    # Strip the single top-level directory prefix from each member name.
    stripped = set()
    for member in members:
        parts = member.split("/", 1)
        if len(parts) > 1:
            stripped.add(parts[1])
        else:
            # Members with no "/" (just filename at root)
            stripped.add(member)

    # Check for sos_commands/ (RHEL sosreport)
    has_sos_commands = any(name.startswith("sos_commands/") for name in stripped)
    if has_sos_commands:
        return "rhel"

    # Check for vmkernel.log (ESXi)
    has_vmkernel = any(name.startswith("var/log/vmkernel") for name in stripped)
    if has_vmkernel:
        return "esxi"

    # Check for commands/ AND var/log/ (ESXi without vmkernel.log)
    has_commands = any(
        name.startswith("commands/") or name == "commands" for name in stripped
    )
    has_var_log = any(name.startswith("var/log/") for name in stripped)
    if has_commands and has_var_log:
        return "esxi"

    # Unrecognized format
    raise ValueError("Cannot detect bundle type: unrecognized archive structure")
