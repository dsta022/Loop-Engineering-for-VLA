"""Reference plugins for ``pipeline/policy_iteration.py``.

Select them with ``module:attribute`` syntax in the iteration config:

- ``policy_backends.command_trainer:create_trainer`` runs any training command
  line and returns the checkpoint it produced.
- ``policy_backends.offline_evaluator:create_evaluator`` scores a checkpoint
  offline by instruction-conditioned action consistency on a held-out dataset.
- ``policy_backends.filesystem_deployer:create_deployer`` publishes a
  checkpoint into a target directory and updates a ``current.json`` pointer.

They are deliberately model-agnostic; swap any of them for a project-specific
plugin that follows the same contract (see docs/plugins.md).
"""
