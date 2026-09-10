# Third-party licences

Licence texts for code **vendored** into this repository — copied into the tree
rather than installed as a dependency. The repository's own licence is the
top-level [`LICENSE`](../LICENSE) (MIT); the files here apply only to the
vendored code they name.

| Vendored code   | Upstream                                                                   | Commit              | Licence                                                               |
|-----------------|----------------------------------------------------------------------------|---------------------|-----------------------------------------------------------------------|
| `alto_tools.py` | [cneud/alto-tools](https://github.com/cneud/alto-tools), Clemens Neudecker | `1f4f01e` (`0.1.0`) | Apache-2.0 — [`alto-tools-Apache-2.0.txt`](alto-tools-Apache-2.0.txt) |

## `alto_tools.py`

Adapted from `src/alto_tools/alto_tools.py` at upstream commit
`1f4f01e5f6ac3562740e39948442973b9cb94be4`. Only the code reachable from the
`alto-tools -t` (text) and `alto-tools -s` (statistics) flags was copied — the
two things this pipeline used. Every change from the original is marked with a
`VENDORED:` comment in the file, as Apache-2.0 §4(b) requires for a derived work;
the module docstring lists what was left behind and why.

Upstream ships no `NOTICE` file, so there is none to reproduce here. The licence
text above is upstream's `LICENSE`, copied verbatim.

Vendored under [issue #50](https://github.com/ufal/atrium-alto-postprocess/issues/50):
the published image used to re-resolve `alto-tools` from a `git+…` requirement,
and the E2E lane then re-installed it from a moving `master` *inside the released
image at run time*. See the "Vendored code" section of the [README](../README.md).
