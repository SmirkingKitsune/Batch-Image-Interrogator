"""Database-vs-sidecar tag analysis shared by both front ends.

The Advanced Image Inspection dialog (ui/dialogs_advanced.py) and the Electron
bridge both need to answer the same questions: which of a model's tags reached
the .txt file, which a filter removed or rewrote, which the user added by hand,
and which tags a set of images has in common. The answers live here so the two
views cannot disagree.

Tag comparisons go through `normalize_tag`, which honours the underscore
setting, so "long_hair" and "long hair" count as the same tag when underscore
replacement is on.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from core.tag_filters import TagFilterSettings

# Threshold used when previewing filters against a stored result. The stored
# row does not carry the threshold it was produced with.
DEFAULT_REVIEW_THRESHOLD = 0.35

WD_RATING_TAGS = (
    ('general', ('rating:safe', 'general', 'rating:general')),
    ('sensitive', ('rating:sensitive', 'sensitive')),
    ('questionable', ('rating:questionable', 'questionable')),
    ('explicit', ('rating:explicit', 'explicit')),
)


def normalize_tag(tag: str, tag_filters: Optional[TagFilterSettings]) -> str:
    """Comparison form of a tag."""
    if tag_filters:
        return tag_filters.normalize_tag_for_comparison(tag)
    return tag.lower()


def build_tag_comparison(
    db_tags: Sequence[str],
    db_confidence: Optional[Dict[str, float]],
    file_tags: Sequence[str],
    tag_filters: Optional[TagFilterSettings],
    threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Compare one model's stored tags against the sidecar file.

    Returns dicts with:
        tag: str (the replacement, when a replace rule applies)
        confidence: Optional[float] (from the database)
        status: in_both | manually_added | db_only | removed_by_filter
                | replaced | file_only
        location: str
        original_tag: Optional[str] (set when replaced)
    """
    db_confidence = db_confidence or {}

    # Apply filters to see what WOULD be written
    if tag_filters and db_confidence:
        filtered_tags, _ = tag_filters.filter_tags_with_confidence(
            list(db_tags), db_confidence, threshold
        )
    else:
        filtered_tags = list(db_tags)

    comparison: List[Dict[str, Any]] = []

    # Normalized lookups handle underscore equivalence.
    file_tags_normalized = {normalize_tag(ft, tag_filters): ft for ft in file_tags}
    filtered_tags_normalized = {normalize_tag(ft, tag_filters) for ft in filtered_tags}

    # Track which file tags have been matched to avoid duplicates
    matched_file_tags_normalized: Set[str] = set()

    for tag in db_tags:
        tag_lower = tag.lower()
        conf = db_confidence.get(tag, 0.0)
        tag_normalized = normalize_tag(tag, tag_filters)

        in_file = tag_normalized in file_tags_normalized
        in_filtered = tag_normalized in filtered_tags_normalized

        if in_file:
            matched_file_tags_normalized.add(tag_normalized)

        if in_file and in_filtered:
            status, location = 'in_both', 'Both'
        elif in_file and not in_filtered:
            status, location = 'manually_added', 'File'
        elif not in_file and in_filtered:
            status, location = 'db_only', 'Database'
        else:
            status, location = 'removed_by_filter', 'Database (filtered)'

        if tag_filters and tag_lower in tag_filters.replace_dict:
            comparison.append({
                'tag': tag_filters.replace_dict[tag_lower],
                'confidence': conf,
                'status': 'replaced',
                'location': location,
                'original_tag': tag,
            })
        else:
            comparison.append({
                'tag': tag,
                'confidence': conf,
                'status': status,
                'location': location,
                'original_tag': None,
            })

    # File-only tags: not produced by this model at all.
    for tag in file_tags:
        if normalize_tag(tag, tag_filters) not in matched_file_tags_normalized:
            comparison.append({
                'tag': tag,
                'confidence': None,
                'status': 'file_only',
                'location': 'File only',
                'original_tag': None,
            })

    return comparison


def filtered_output_tags(
    db_tags: Sequence[str],
    db_confidence: Optional[Dict[str, float]],
    tag_filters: Optional[TagFilterSettings],
    threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> List[str]:
    """The tags a batch write would produce from a stored result."""
    if not tag_filters:
        return list(db_tags)
    if db_confidence:
        filtered, _ = tag_filters.filter_tags_with_confidence(list(db_tags), db_confidence, threshold)
        return filtered
    return tag_filters.apply_filters(list(db_tags))


def plan_apply_to_file(
    db_tags: Sequence[str],
    db_confidence: Optional[Dict[str, float]],
    file_tags: Sequence[str],
    tag_filters: Optional[TagFilterSettings],
    threshold: float = DEFAULT_REVIEW_THRESHOLD,
) -> Dict[str, Any]:
    """What writing a stored result into the sidecar would change.

    Existing sidecar tags keep their order. A tag with a replace rule is
    rewritten in place, filtered tags missing from the file are appended, and
    tags only the file carries (manual additions, prefixes) are kept.

    Returns {"tags", "added", "rewritten": [(old, new)], "kept_manual"}.
    """
    output = filtered_output_tags(db_tags, db_confidence, tag_filters, threshold)
    replace_dict = dict(tag_filters.replace_dict) if tag_filters else {}
    db_normalized = {normalize_tag(tag, tag_filters) for tag in db_tags}

    result: List[str] = []
    seen: Set[str] = set()
    rewritten: List[Tuple[str, str]] = []
    kept_manual: List[str] = []

    for tag in file_tags:
        replacement = replace_dict.get(tag.lower())
        final = replacement if replacement else tag
        normalized = normalize_tag(final, tag_filters)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(final)
        if replacement and replacement != tag:
            rewritten.append((tag, replacement))
        elif normalize_tag(tag, tag_filters) not in db_normalized:
            kept_manual.append(tag)

    added: List[str] = []
    for tag in output:
        normalized = normalize_tag(tag, tag_filters)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(tag)
        added.append(tag)

    return {
        "tags": result,
        "added": added,
        "rewritten": rewritten,
        "kept_manual": kept_manual,
    }


def extract_wd_ratings(tags: Sequence[str], confidence_scores: Dict[str, float]) -> Dict[str, float]:
    """WD sensitivity ratings from a result's tags.

    Returns general / sensitive / questionable / explicit confidences (0.0-1.0).
    """
    ratings = {name: 0.0 for name, _aliases in WD_RATING_TAGS}
    for rating_name, possible_tags in WD_RATING_TAGS:
        for tag in possible_tags:
            if tag in tags and tag in confidence_scores:
                ratings[rating_name] = confidence_scores[tag]
                break
    return ratings


def collect_editor_tags(
    interrogations: Iterable[Dict[str, Any]],
    file_tags: Sequence[str],
    tag_filters: Optional[TagFilterSettings],
) -> Tuple[List[str], List[str]]:
    """Every tag worth offering in a checkbox editor, and which are on disk.

    Model tags come first in canonical form (the database spelling wins over
    the file's), then file-only tags. Returns (all_tags, selected_tags), both
    sorted case-insensitively.
    """
    canonical: Dict[str, str] = {}
    for interrog in interrogations:
        for tag in interrog.get('tags', []) or []:
            canonical.setdefault(normalize_tag(tag, tag_filters), tag)
    for tag in file_tags or []:
        canonical.setdefault(normalize_tag(tag, tag_filters), tag)

    file_normalized = {normalize_tag(tag, tag_filters) for tag in file_tags or []}
    all_tags = sorted(canonical.values(), key=str.lower)
    selected = [tag for tag in all_tags if normalize_tag(tag, tag_filters) in file_normalized]
    return all_tags, selected


def compute_common_tags(
    per_image_tags: Iterable[Tuple[Iterable[Sequence[str]], Sequence[str]]],
    tag_filters: Optional[TagFilterSettings],
) -> Set[str]:
    """Tags present on every image, in canonical form.

    `per_image_tags` yields (database tag lists, sidecar tags) per image. A tag
    counts for an image when either source has it. The database spelling is
    preferred as the canonical form, then the first spelling encountered.
    """
    normalized_tag_sets: List[Set[str]] = []
    canonical_forms: Dict[str, str] = {}

    for db_tag_lists, file_tags in per_image_tags:
        image_normalized: Set[str] = set()
        for tags in db_tag_lists:
            for tag in tags or []:
                normalized = normalize_tag(tag, tag_filters)
                image_normalized.add(normalized)
                canonical_forms.setdefault(normalized, tag)
        for tag in file_tags or []:
            normalized = normalize_tag(tag, tag_filters)
            image_normalized.add(normalized)
            canonical_forms.setdefault(normalized, tag)
        normalized_tag_sets.append(image_normalized)

    if not normalized_tag_sets:
        return set()
    common = set.intersection(*normalized_tag_sets)
    return {canonical_forms[n] for n in common if n in canonical_forms}


def apply_common_tag_edits(
    current_tags: Sequence[str],
    tags_to_remove: Iterable[str],
    tags_to_add: Iterable[str],
    tag_filters: Optional[TagFilterSettings],
) -> List[str]:
    """One image's sidecar after a multi-image common-tag edit.

    Removals match by normalized form, so removing "long_hair" also removes
    "long hair". Additions are skipped when an equivalent tag is present. Tags
    unique to the image are preserved in their original order.
    """
    remove_normalized = {normalize_tag(tag, tag_filters) for tag in tags_to_remove}
    new_tags: List[str] = []
    present: Set[str] = set()
    for tag in current_tags:
        normalized = normalize_tag(tag, tag_filters)
        if normalized not in remove_normalized:
            new_tags.append(tag)
            present.add(normalized)
    for tag in tags_to_add:
        normalized = normalize_tag(tag, tag_filters)
        if normalized not in present:
            new_tags.append(tag)
            present.add(normalized)
    return new_tags
