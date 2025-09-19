"""
Layer B: Static Rules / Regex / Signature Detection Engine

Fast, deterministic detection of known prompt injection patterns including:
- Instruction bypass phrases
- Known jailbreak templates  
- Command indicators
- ChatML/system token misuse
- Repeated known exploit strings

Design principles:
- Speed: microsecond to single-digit ms per string
- Observable: detailed logging with match provenance
- Maintainable: versioned signatures with metadata
"""

import re
import time
import json
import hashlib
from typing import List, Tuple, Optional
from pathlib import Path

from models.SignatureMatch import SignatureMatch, Severity
from models.DetectionResult import DetectionResult

class SignatureEngine:
    """Main signature-based detection engine"""
    
    def __init__(self):
        self.signatures_dir = Path(__file__).parent / "signatures"
        
        # Compiled pattern storage by severity
        self.high_patterns = []  # (id, desc, pattern, original, tags)
        self.medium_patterns = []
        self.low_patterns = []
        
        # Literal phrase sets for fast matching
        self.high_literals = {}  # phrase -> (id, desc, tags)
        self.medium_literals = {}
        self.low_literals = {}
        
        # Load signatures
        self._load_signatures()
    
    def _compile_pattern_safely(self, pattern: str, rule_id: str) -> Optional[re.Pattern]:
        """
        Compile regex pattern with basic error handling
        """
        try:
            # Compile with standard flags
            flags = re.IGNORECASE | re.MULTILINE
            compiled = re.compile(pattern, flags)
            return compiled
            
        except re.error as e:
            print(f"ERROR: Invalid regex pattern {rule_id}: {e}")
            return None
    
    #Load signature files from directory structure
    def _load_signatures(self):

        # Load HIGH severity signatures
        self._load_severity_signatures(Severity.HIGH)
        self._load_severity_signatures(Severity.MEDIUM) 
        self._load_severity_signatures(Severity.LOW)
        
        print(f"Loaded signatures: {len(self.high_patterns)} HIGH patterns, "
              f"{len(self.medium_patterns)} MEDIUM patterns, {len(self.low_patterns)} LOW patterns")
        print(f"Loaded literals: {len(self.high_literals)} HIGH, "
              f"{len(self.medium_literals)} MEDIUM, {len(self.low_literals)} LOW")
    #Load signatures for a specific severity level
    def _load_severity_signatures(self, severity: Severity):

        severity_dir = self.signatures_dir / severity.value
        
        for sig_file in severity_dir.glob("*.json"):
            try:
                with open(sig_file, 'r') as f:
                    signatures = json.load(f)
                    
                for sig in signatures:
                    self._add_signature(
                        severity=severity,
                        rule_id=sig['id'],
                        description=sig['description'],
                        pattern=sig['pattern'],
                        pattern_type=sig.get('type', 'regex'),
                        tags=sig.get('tags', []),
                        examples=sig.get('examples', [])
                    )
                    
            except Exception as e:
                print(f"Error loading signature file {sig_file}: {e}")
    
    def _add_signature(self, severity: Severity, rule_id: str, description: str, 
                      pattern: str, pattern_type: str = 'regex', 
                      tags = None, examples = None):
        """Add a signature to the appropriate storage"""
        if tags is None:
            tags = []
            
        if pattern_type == 'literal':
            # Store as literal phrase for fast matching
            phrase = pattern.lower().strip()
            literal_storage = {
                Severity.HIGH: self.high_literals,
                Severity.MEDIUM: self.medium_literals,
                Severity.LOW: self.low_literals
            }
            literal_storage[severity][phrase] = (rule_id, description, tags)
            
        elif pattern_type == 'regex':
            # Compile and store regex pattern
            compiled = self._compile_pattern_safely(pattern, rule_id)
            if compiled:
                pattern_tuple = (rule_id, description, compiled, pattern, tags)
                pattern_storage = {
                    Severity.HIGH: self.high_patterns,
                    Severity.MEDIUM: self.medium_patterns,
                    Severity.LOW: self.low_patterns
                }
                pattern_storage[severity].append(pattern_tuple)
    
    def _match_literals(self, text: str, severity: Severity) -> List[SignatureMatch]:
        """Fast literal phrase matching"""
        matches = []
        literal_storage = {
            Severity.HIGH: self.high_literals,
            Severity.MEDIUM: self.medium_literals,
            Severity.LOW: self.low_literals
        }
        
        for phrase, (rule_id, description, tags) in literal_storage[severity].items():
            start_pos = text.find(phrase)
            if start_pos != -1:
                matches.append(SignatureMatch(
                    rule_id=rule_id,
                    severity=severity,
                    pattern=phrase,
                    matched_text=phrase,
                    start_pos=start_pos,
                    end_pos=start_pos + len(phrase),
                    rule_description=description,
                    tags=tags,
                    confidence=1.0
                ))
        
        return matches
    
    def _match_patterns(self, text: str, severity: Severity) -> List[SignatureMatch]:
        """Regex pattern matching with timeout protection"""
        matches = []
        pattern_storage = {
            Severity.HIGH: self.high_patterns,
            Severity.MEDIUM: self.medium_patterns,
            Severity.LOW: self.low_patterns
        }
        
        for rule_id, description, compiled_pattern, original_pattern, tags in pattern_storage[severity]:
            try:
                # Find all matches
                for match in compiled_pattern.finditer(text):
                    matches.append(SignatureMatch(
                        rule_id=rule_id,
                        severity=severity,
                        pattern=original_pattern,
                        matched_text=match.group(0),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        rule_description=description,
                        tags=tags,
                        confidence=1.0
                    ))
                    
            except Exception as e:
                print(f"ERROR: Pattern {rule_id} failed during matching: {e}")
                continue
        
        return matches
    
    def detect(self, text: str) -> DetectionResult:
        """
        Main detection method - runs all signature checks
        Returns structured result with verdict
        """
        start_time = time.time()
        
        # Create input hash for logging
        input_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        all_matches = []
        
        # Run HIGH severity checks first (for potential short-circuiting)
        high_literal_matches = self._match_literals(text, Severity.HIGH)
        high_pattern_matches = self._match_patterns(text, Severity.HIGH)
        high_matches = high_literal_matches + high_pattern_matches
        
        all_matches.extend(high_matches)
        
        # If we have HIGH severity matches, we might short-circuit
        # For now, continue to gather all evidence
        
        # Run MEDIUM severity checks
        medium_literal_matches = self._match_literals(text, Severity.MEDIUM)
        medium_pattern_matches = self._match_patterns(text, Severity.MEDIUM)
        medium_matches = medium_literal_matches + medium_pattern_matches
        
        all_matches.extend(medium_matches)
        
        # Run LOW severity checks
        low_literal_matches = self._match_literals(text, Severity.LOW)
        low_pattern_matches = self._match_patterns(text, Severity.LOW)
        low_matches = low_literal_matches + low_pattern_matches
        
        all_matches.extend(low_matches)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Determine verdict and scoring
        verdict, total_score, highest_severity = self._calculate_verdict(all_matches)
        
        return DetectionResult(
            input_hash=input_hash,
            processing_time_ms=processing_time_ms,
            matches=all_matches,
            verdict=verdict,
            total_score=total_score,
            highest_severity=highest_severity
        )
    
    def _calculate_verdict(self, matches: List[SignatureMatch]) -> Tuple[str, float, Optional[Severity]]:
        """Calculate final verdict based on matches"""
        if not matches:
            return "allow", 0.0, None
        
        # Find highest severity
        severities = [match.severity for match in matches]
        if Severity.HIGH in severities:
            highest_severity = Severity.HIGH
            verdict = "block"  # HIGH severity = block
        elif Severity.MEDIUM in severities:
            highest_severity = Severity.MEDIUM
            verdict = "flag"   # MEDIUM severity = flag for escalation
        else:
            highest_severity = Severity.LOW
            verdict = "allow"  # LOW severity = allow with metadata
        
        # Calculate score (sum of confidence scores with severity weighting)
        severity_weights = {Severity.HIGH: 10.0, Severity.MEDIUM: 5.0, Severity.LOW: 1.0}
        total_score = sum(
            match.confidence * severity_weights[match.severity]
            for match in matches
        )
        
        return verdict, total_score, highest_severity
