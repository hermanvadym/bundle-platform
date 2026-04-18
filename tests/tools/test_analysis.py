from bundle_platform.tools.analysis import grep_log


def test_grep_log_finds_match(tmp_path):
    log = tmp_path / "messages"
    log.write_text("Apr 15 02:31:00 host kernel: oom_kill process\nother line\n")
    result = grep_log(str(tmp_path), "messages", r"oom_kill", context_lines=0)
    assert "oom_kill" in result


def test_grep_log_no_match(tmp_path):
    log = tmp_path / "messages"
    log.write_text("normal line\n")
    result = grep_log(str(tmp_path), "messages", r"oom_kill", context_lines=0)
    assert "no matches" in result.lower() or result.strip() == ""


def test_grep_log_missing_file(tmp_path):
    result = grep_log(str(tmp_path), "nonexistent.log", r"anything", context_lines=0)
    assert "not found" in result.lower() or "error" in result.lower()
