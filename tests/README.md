# AstraTensor tests

Run the core autograd tests with:

    python -m pip install -e .
    python -m pip install pytest
    python -m pytest -q

The tests cover scalar and matrix operations, reverse-mode gradient propagation,
shared computation paths, scalar broadcasting, shape validation, and gradient
accumulation across repeated backward calls.
