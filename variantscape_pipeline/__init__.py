"""Re-runnable Variantscape pipeline.

Consolidates the notebooks in ``notebooks/01`` - ``notebooks/06`` into an
incremental pipeline that keeps its state in a SQLite database and produces the
artifacts consumed by EvidenceDb:

- ``network_graph_weighted.gml``
- ``final_variant_treatment_consensus.csv``
- ``metadata_mapping_transposed.csv``

Run ``python variantscape_pipeline/pipeline.py --help`` (or ``python -m variantscape_pipeline --help``) for usage.
"""

__version__ = "1.0.0"
