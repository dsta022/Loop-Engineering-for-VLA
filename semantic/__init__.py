"""Concrete semantic evaluator backends for the quality audit.

The base audit and its model-free consistency checks never import this package.
It is loaded only when an audit is invoked with an explicit evaluator spec, for
example::

    --semantic-evaluator semantic.clip_alignment_evaluator:create_evaluator

The heavy vision-language dependencies (torch, open_clip) are imported lazily
inside the backend classes, so importing this package, running the base audit,
or running the unit tests never requires a model runtime.
"""
