from bundle_platform.shared.caps import cap_lines


def test_cap_lines_under_limit():
    text = "\n".join(f"line {i}" for i in range(5))
    assert cap_lines(text, 10) == text


def test_cap_lines_over_limit():
    text = "\n".join(f"line {i}" for i in range(20))
    result = cap_lines(text, 5)
    lines = result.splitlines()
    assert len(lines) == 6  # 5 kept + marker
    assert "truncated" in lines[-1]
    assert "showing 5 of 20" in lines[-1]


def test_cap_lines_empty():
    assert cap_lines("", 10) == ""
