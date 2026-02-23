"""Hold-out validation filter for extracted YARA signatures.

After extracting candidate signatures from the training split, we compile
them into a temporary YARA rule-set and scan a held-out portion of safe
text.  Any rule that fires on *any* held-out safe document is discarded.

This catches common-English patterns that achieved high training precision
only because the training safe distribution didn't happen to contain them.
"""

import logging
import tempfile
from pathlib import Path

import yara

from core.layer_b.extraction.yara_writer import write_yara_rules

log = logging.getLogger(__name__)


def filter_signatures_with_holdout(signatures, holdout_safe_texts, rule_prefix, meta_keys,*, max_safe_hits: int = 0):
    """Remove signatures that fire on held-out safe text."""
    if not signatures or not holdout_safe_texts:
        return signatures

    # Write candidates to a temporary YARA file so we can compile and scan.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_yar = Path(tmpdir) / "candidates.yar"
        write_yara_rules(tmp_yar, rule_prefix, signatures, meta_keys)

        rules = yara.compile(filepath=str(tmp_yar))

        # Build a mapping: rule_name -> index
        from core.layer_b.extraction.yara_writer import _pattern_digest

        name_to_idx = {}
        for idx, sig in enumerate(signatures):
            pattern = str(sig["pattern"]).strip()
            if not pattern:
                continue
            digest = _pattern_digest(pattern)
            rule_name = f"{rule_prefix}{digest}_{idx + 1:04d}"
            name_to_idx[rule_name] = idx

        # Scan every held-out safe document and record which rules fire.
        fired_on_safe = {}
        for text in holdout_safe_texts:
            matches = rules.match(data=text)
            for m in matches:
                fired_on_safe[m.rule] = fired_on_safe.get(m.rule, 0) + 1

        # Identify bad indices
        bad_indices = set()
        for rule_name, count in fired_on_safe.items():
            if count > max_safe_hits:
                idx = name_to_idx.get(rule_name)
                if idx is not None:
                    bad_indices.add(idx)

        kept = [sig for i, sig in enumerate(signatures) if i not in bad_indices]

        log.info(
            "Holdout filter: %d / %d signatures removed (fired on safe holdout); %d kept",
            len(bad_indices),
            len(signatures),
            len(kept),
        )
        return kept
