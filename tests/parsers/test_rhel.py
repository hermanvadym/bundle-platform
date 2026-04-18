from bundle_platform.parsers.rhel import get_adapter


def test_get_adapter_returns_bundle_adapter():
    from bundle_platform.parsers.base import BundleAdapter
    adapter = get_adapter()
    assert isinstance(adapter, BundleAdapter)


def test_bundle_type():
    assert get_adapter().bundle_type == "rhel"


def test_tag_file_system_logs():
    adapter = get_adapter()
    assert adapter.tag_file("var/log/messages") == "system_logs"


def test_tag_file_sos_commands():
    adapter = get_adapter()
    assert adapter.tag_file("sos_commands/uname") == "sos_commands"


def test_error_sweep_categories():
    adapter = get_adapter()
    cats = adapter.error_sweep_categories()
    assert "system_logs" in cats


def test_failure_patterns_stubbed():
    assert get_adapter().failure_patterns() == ""
