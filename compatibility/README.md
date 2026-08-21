# Compatibility layer

The historical `autotrack.*` imports remain available through
`compatibility/autotrack` while implementation ownership moves to the unique
method namespaces. The old dated Hybrid project,
`web/`, and short method aliases under `compatibility/methods/` point to their
new locations;
legacy virtual environments are preserved under `archive/legacy_envs/`.

New code should import neutral data/protocol helpers from
`auto_track_common` and method code from its project namespace.
