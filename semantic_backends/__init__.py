"""Built-in semantic evaluator backends for the quality audit.

Each module here is a self-contained ``--semantic-evaluator`` plugin selected
with ``module:attribute`` syntax, e.g.::

    --semantic-evaluator semantic_backends.vla_checkpoint_evaluator:create_evaluator

Heavy dependencies (torch, lerobot) are imported lazily inside the factory, so
importing this package never pulls model runtimes into the base audit.
"""
