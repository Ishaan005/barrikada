"""
Tool Document representation for ToolHijacker attack generator.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolDocument:
    """
    Represents a tool document with name and description.
    The malicious tool document is denoted as d_t in the paper.
    """
    name: str
    description: str
    metadata: dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"Tool: {self.name}\nDescription: {self.description}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'description': self.description,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ToolDocument':
        """Create ToolDocument from dictionary"""
        return cls(
            name=data['name'],
            description=data['description'],
            metadata=data.get('metadata', {})
        )


@dataclass
class MaliciousToolDocument(ToolDocument):
    """
    Malicious tool document with decomposed description.
    The description is split into R (retrieval) and S (selection) subsequences.
    """
    retrieval_subsequence: Optional[str] = None  # R in the paper
    selection_subsequence: Optional[str] = None  # S in the paper
    
    def compose_description(self) -> str:
        """
        Compose full description from R ⊕ S (concatenation).
        """
        parts = []
        if self.retrieval_subsequence:
            parts.append(self.retrieval_subsequence)
        if self.selection_subsequence:
            parts.append(self.selection_subsequence)
        
        if parts:
            return " ".join(parts)
        return self.description if self.description else ""
    
    def update_from_subsequences(self):
        """Update the main description from R and S"""
        self.description = self.compose_description()
    
    def set_retrieval_subsequence(self, r: str):
        """Set R and update description"""
        self.retrieval_subsequence = r
        self.update_from_subsequences()
    
    def set_selection_subsequence(self, s: str):
        """Set S and update description"""
        self.selection_subsequence = s
        self.update_from_subsequences()
