"""Hold-out validation filter for extracted YARA signatures.

After extracting candidate signatures from the training split, we compile
them into a temporary YARA rule-set and scan held-out safe text. Any rule
that fires too often on safe text is discarded — it's probably matching
common English that just happened to be absent from the training set.
"""
import logging
import tempfile
from pathlib import Path

import yara

from core.layer_b.extraction.yara_writer import write_yara_rules

log = logging.getLogger(__name__)


def filter_signatures_with_holdout(signatures, holdout_safe_texts, rule_prefix, meta_keys, *, max_safe_hits=0):
    """Remove signatures that fire on held-out safe text.

    Compiles all candidate signatures into a temp YARA file, scans each
    held-out safe document, and drops any rule that fires more than
    max_safe_hits times.
    """
    if not signatures or not holdout_safe_texts:
        return signatures

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_yar = Path(tmpdir) / "candidates.yar"
        write_yara_rules(tmp_yar, rule_prefix, signatures, meta_keys)
        rules = yara.compile(filepath=str(tmp_yar))

        # We need to map YARA rule names back to signature indices so we
        # know which ones to remove. The rule names are built the same way
        # as in write_yara_rules: "{prefix}{digest}_{index:04d}"
        from core.layer_b.extraction.yara_writer import _pattern_digest

        name_to_index = {}
        for idx, sig in enumerate(signatures):
            pattern = str(sig["pattern"]).strip()
            if not pattern:
                continue
            digest = _pattern_digest(pattern)
            rule_name = f"{rule_prefix}{digest}_{idx + 1:04d}"
            name_to_index[rule_name] = idx

        # Scan every held-out safe doc and count how often each rule fires
        safe_hit_counts = {}
        for text in holdout_safe_texts:
            for match in rules.match(data=text):
                safe_hit_counts[match.rule] = safe_hit_counts.get(match.rule, 0) + 1

        # Find which signatures fired too often on safe text
        indices_to_remove = set()
        for rule_name, count in safe_hit_counts.items():
            if count > max_safe_hits:
                idx = name_to_index.get(rule_name)
                if idx is not None:
                    indices_to_remove.add(idx)

        kept = [sig for i, sig in enumerate(signatures) if i not in indices_to_remove]

        log.info(
            "Holdout filter: %d / %d signatures removed (fired on safe holdout); %d kept",
            len(indices_to_remove), len(signatures), len(kept),
        )
        return kept
