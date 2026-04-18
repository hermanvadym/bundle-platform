from bundle_platform.parsers import esxi, rhel
from bundle_platform.parsers.base import BundleAdapter
from bundle_platform.parsers.detect import detect_bundle_type

__all__ = ["BundleAdapter", "detect_bundle_type", "load_adapter"]


def load_adapter(bundle_type: str) -> BundleAdapter:
    """Return the BundleAdapter for the given bundle type string."""
    if bundle_type == "rhel":
        return rhel.get_adapter()
    if bundle_type == "esxi":
        return esxi.get_adapter()
    raise ValueError(f"unknown bundle type: {bundle_type!r}")
