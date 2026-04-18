import io
import tarfile
from pathlib import Path

import pytest

from bundle_platform.parsers.detect import detect_bundle_type


def _make_tar(tmp: Path, members: list[str]) -> Path:
    bundle = tmp / "test.tar.gz"
    with tarfile.open(bundle, "w:gz") as tf:
        for name in members:
            info = tarfile.TarInfo(name=name)
            info.size = 0
            tf.addfile(info, io.BytesIO(b""))
    return bundle


def test_detects_rhel(tmp_path):
    bundle = _make_tar(tmp_path, ["sosreport-host/sos_commands/uname"])
    assert detect_bundle_type(bundle) == "rhel"


def test_detects_esxi(tmp_path):
    bundle = _make_tar(tmp_path, ["esx-host/commands/esxcfg-info", "esx-host/var/log/vmkernel.log"])
    assert detect_bundle_type(bundle) == "esxi"


def test_raises_on_unknown(tmp_path):
    bundle = _make_tar(tmp_path, ["random/file.txt"])
    with pytest.raises(ValueError, match="Cannot detect"):
        detect_bundle_type(bundle)
