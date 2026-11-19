"""Your Pydantic schema for one triaged customer message (task 3).

Write your own schema, not a copy of Routing from the demo. The assignment lists
what it must contain; your variant adds its own requirement.
"""

from pydantic import BaseModel


class Triage(BaseModel):
    """Triage of one customer message."""

    # TODO (task 3): fields with types, allowed values, limits and descriptions
