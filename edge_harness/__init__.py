"""Shared measurement infrastructure for the edge-deploy labs.

The API mirrors ``mlsys_harness`` in the sibling mlsys-infra repository
(``timing.summarize``, ``writer.JsonlWriter``, ``claims.check_spec``), so a lab
can move between the two repositories without rewriting its bookkeeping. This
package adds what the edge labs need and the GPU labs do not: building and
running small C++ benchmarks with g++, and the edge-device datasheet table.

Importing this package does not import torch.
"""

__version__ = "1"
