"""Shared domain models for Forge."""


# ===================================================================
#                           Data Models
# ===================================================================
class Process:
    """Represents a single recipe with its inputs and outputs."""
    __slots__ = ('name', 'inputs', 'outputs')
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = dict(inputs or {})
        self.outputs = dict(outputs or {})
