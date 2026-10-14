# SPDX-License-Identifier: Apache-2.0

import getpass
import os
import tempfile

import psutil

from areal.utils.logging import getLogger

logger = getLogger("FileSystemUtils")

# Keywords to match in fstype or device path (case-insensitive substring match)
NETWORK_FS_KEYWORDS = {
    # Standard network filesystems
    "nfs",
    "lustre",
    "gpfs",
    "mmfs",
    "beegfs",
    "pvfs",
    "orangefs",
    "ceph",
    "glusterfs",
    "afs",
    "cifs",
    "smb",
    "rbd",
    "9p",
    # Cloud providers
    "alinas",
    "cpfs",
    "vepfs",
    "goosefs",
    "chdfs",
    "obsfs",
    "gcsfuse",
    "efs",
    "blobfuse",
    "s3fs",
    "fsx",
    # Distributed filesystems
    "juicefs",
    "alluxio",
    "seaweedfs",
    "hdfs",
    "moosefs",
    "lizardfs",
    "xtreemfs",
    "sshfs",
}


def get_user_tmp():
    """Use an explicit cache root, XDG cache, or a per-user temporary directory."""
    user_tmp = os.environ.get("AREAL_CACHE_DIR")
    if not user_tmp:
        cache_root = os.environ.get("XDG_CACHE_HOME")
        user_tmp = (
            os.path.join(cache_root, "areal")
            if cache_root
            else os.path.join(tempfile.gettempdir(), f"areal-{getpass.getuser()}")
        )
    user_tmp = os.path.expanduser(user_tmp)
    os.makedirs(user_tmp, exist_ok=True)
    return user_tmp


def validate_shared_path(path: str, name: str = "path", warn_nfs: bool = True) -> None:
    """
    Validate that a path exists for shared/distributed access.

    Args:
        path: Path to validate.
        name: Descriptive name for the path (used in messages).
        warn_nfs: If True, warn when not on network filesystem.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not path:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} '{path}' does not exist.")

    if not warn_nfs:
        return

    # Check if path is on network filesystem
    real_path = os.path.realpath(path)
    partitions = psutil.disk_partitions(all=True)

    # Find the best matching mountpoint (longest prefix match)
    best_match = None
    best_match_len = 0

    for partition in partitions:
        mountpoint = partition.mountpoint
        if real_path.startswith(mountpoint) and len(mountpoint) > best_match_len:
            best_match = partition
            best_match_len = len(mountpoint)

    is_network = False
    if best_match:
        fstype = best_match.fstype.lower()
        device = best_match.device.lower()
        combined = fstype + device

        is_network = (
            any(kw in combined for kw in NETWORK_FS_KEYWORDS)
            or (":" in device and not device.startswith("/dev"))
            or device.startswith("//")
        )

    if not is_network:
        logger.warning(
            f"{name} '{path}' is not on a network filesystem. "
            "This may cause issues in distributed training where all nodes "
            "need access to the same files. Consider using NFS, Lustre, or "
            "other shared storage."
        )
