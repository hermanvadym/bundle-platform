from bundle_platform.parsers.esxi import get_adapter


def test_get_adapter_returns_bundle_adapter():
    from bundle_platform.parsers.base import BundleAdapter
    adapter = get_adapter()
    assert isinstance(adapter, BundleAdapter)


def test_bundle_type():
    assert get_adapter().bundle_type == "esxi"


def test_tag_file_system_logs():
    adapter = get_adapter()
    assert adapter.tag_file("var/log/vmkernel.log") == "system_logs"


def test_tag_file_commands():
    adapter = get_adapter()
    assert adapter.tag_file("commands/esxcfg-info") == "commands"


def test_failure_patterns_stubbed():
    assert get_adapter().failure_patterns() == ""
