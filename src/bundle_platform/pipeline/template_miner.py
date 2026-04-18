# src/bundle_platform/pipeline/template_miner.py
from __future__ import annotations


class TemplateMinerWrapper:
    """Thin wrapper around the drain3 TemplateMiner for log template extraction.

    Drain3 is a streaming log parser that groups similar log lines into templates
    by replacing variable tokens (IPs, PIDs, device IDs, timestamps) with <*>.

    Why use it?
    Embedding the template "SCSI error on device <*>" instead of
    "SCSI error on device naa.600a098..." groups all similar SCSI errors into
    the same region of vector space, improving recall for generic questions like
    "were there any storage errors?".

    Why be careful with it?
    When the variable IS the evidence (e.g., the exact device ID that failed),
    Drain3 destroys it. Monitor evidence_regex_match in the scorecard — if it
    drops vs baseline, add the file category to an exclusion list.
    """

    def __init__(self) -> None:
        from drain3 import TemplateMiner
        self._miner = TemplateMiner()

    def add_log_message(self, message: str) -> str:
        """Parse a log line and return its template string.

        Returns the template with variables replaced by <*>.
        If drain3 cannot parse the line, returns the original message unchanged.
        """
        result = self._miner.add_log_message(message)
        if result and result.get("template_mined"):
            return result["template_mined"]
        return message
