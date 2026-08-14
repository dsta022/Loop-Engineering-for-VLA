"""Toolkit test package.

This file must exist: it makes ``tests`` a regular package that always wins
over same-named top-level ``tests`` packages that some third-party
distributions (e.g. draccus) install into site-packages, which would otherwise
shadow the implicit namespace package and break ``from tests.<module> import``
inside the suite.
"""
