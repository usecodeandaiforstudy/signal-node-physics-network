"""
extensions -- work that builds ON the base version, never inside it.

Scope rule (see multigraph/DESIGN.md): everything here depends on the base
version READ-ONLY.  Nothing in this package edits, monkey-patches or subclasses
anything under experiments/, substrate_diffusion/, simulator/, network/ or
training/.  If a base change ever looks necessary, that is a finding to report.
"""
