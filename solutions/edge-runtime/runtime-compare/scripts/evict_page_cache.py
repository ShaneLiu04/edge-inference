"""Drop the page cache of files under the given directories (posix_fadvise DONTNEED).

The AutoDL no-GPU container caps the cgroup at 2 GB and counts page cache against it. A long run
(pip installs, TVM tuning logs) fills that cache, and processes then die with SIGKILL while the
kernel's oom_kill counter stays 0. Neither /proc/sys/vm/drop_caches nor memory.reclaim is
writable there, so the cache is dropped file by file between runs.

Usage: python evict_page_cache.py DIR [DIR ...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def cgroup_bytes() -> int | None:
    p = Path("/sys/fs/cgroup/memory.current")
    return int(p.read_text()) if p.exists() else None


def evict(roots: list[str]) -> int:
    files = 0
    for root in roots:
        for dirpath, _, names in os.walk(root):
            for name in names:
                try:
                    fd = os.open(os.path.join(dirpath, name), os.O_RDONLY | os.O_NOFOLLOW)
                except OSError:
                    continue
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    files += 1
                except OSError:
                    pass
                finally:
                    os.close(fd)
    return files


def main() -> None:
    before = cgroup_bytes()
    files = evict(sys.argv[1:])
    after = cgroup_bytes()
    if before is not None and after is not None:
        print(f"evicted {files} files: cgroup {before / 2**20:.0f} -> {after / 2**20:.0f} MiB")
    else:
        print(f"evicted {files} files")


if __name__ == "__main__":
    main()
