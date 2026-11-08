from abc import ABC, abstractmethod
from typing import Dict, Tuple

from ..utils.misc_utils import NerRawOutput, TripleRawOutput


class BaseOpenIE(ABC):
    """Lifecycle and execution contract shared by OpenIE implementations."""

    def __init__(self, active: bool = True) -> None:
        self.active = active

    def _require_active(self) -> None:
        if not self.active:
            raise RuntimeError(
                f"{self.__class__.__name__} is disabled. Enable it before extracting information."
            )

    def disable(self) -> None:
        """Disable extraction and release implementation-owned resources."""
        if self.active:
            self.close()
        self.active = False

    def enable(self) -> None:
        """Mark extraction as available after an explicit disable."""
        self.active = True

    def close(self) -> None:
        """Release implementation-owned resources."""
        self.active = False

    @abstractmethod
    def batch_openie(
        self, chunks: Dict[str, dict]
    ) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """Extract entities and triples for a batch of chunks."""
        raise NotImplementedError
