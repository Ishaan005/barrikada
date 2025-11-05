"""
Base interface for layer results to ensure consistency across all layers.
All layer result classes should follow this pattern.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class LayerResult(ABC):
    """Abstract base class for layer results"""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        pass
    
    @abstractmethod
    def get_risk_score(self) -> float:
        """
        Get risk score contribution from this layer (0-100)
        Used by orchestrator for final risk aggregation
        """
        pass
    
    @property
    @abstractmethod
    def verdict(self) -> str:
        """Get layer verdict: 'allow', 'flag', or 'block'"""
        pass
    
    @property
    @abstractmethod
    def processing_time_ms(self) -> float:
        """Get processing time in milliseconds"""
        pass
    
    @property
    def confidence_score(self) -> float:
        """
        Get confidence score (0.0 to 1.0)
        Default implementation returns 1.0, override if layer provides confidence
        """
        return 1.0
