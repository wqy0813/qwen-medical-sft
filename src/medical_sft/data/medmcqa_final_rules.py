"""Frozen, independent MedMCQA rules. Standard library only; no legacy classifiers.

All detector matches retain field/offset evidence. Only explicit content rules
affect quality; weak hints never affect the class or training eligibility.
"""
from __future__ import annotations

import copy
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from html.parser import HTMLParser

from medical_sft.data.medmcqa_answers import parse_answer

VERSION = "medmcqa-final-20261031"
LETTERS = "ABCD"
PLACEHOLDERS = frozenset({
    ".", "-", "--", "none", "null", "n/a", "na", "no explanation", "not available",
    "explanation not available here", "no explanation provided", "no explanation available", "not provided",
})
DECLARATION = re.compile(
    r"^\s*(?:(?:the\s+correct\s+answer\s+is|correct\s+answer|ans(?:wer)?)[.:]?\s*"
    r"(?:is\s+)?[:=]?\s*\(?([A-D])\)?(?=$|[\s.):;,\-])\s*[):;,\-\.]?\s*|"
    r"([A-D])\s+is\s+correct[.!]?\s*$)", re.I)
REFERENCE_HEADER = re.compile(r"^\s*(?:references?|bibliography|sources?|refs?)\s*:\s*(.*)$", re.I)
REFERENCE_TITLE = re.compile(r"^\s*(?:references?|bibliography|sources?|refs?)\s*$", re.I)
EDITION_ONLY = re.compile(r"^\d+(?:st|nd|rd|th)\s+(?:edition|ed\.)[.,]?\s*$", re.I)
CITATION = re.compile(
    r"^(?:\[\d+\]\s+.+|\d+[.)]\s+.+|(?:doi|isbn)\s*:\s*\S.*|https?://doi\.org/\S+|"
    r".+\b\d+(?:st|nd|rd|th)\s+(?:edition|ed\.)(?:[,.]?\s*(?:pp?\.?\s*\d+(?:[-–]\d+)?))?[.,]?)$", re.I)
IMAGE_RESOURCE = re.compile(
    r"<\s*(?:img|svg|picture)\b|!\[[^\]\n]*\]\([^\n]+\)|!\[[^\]\n]*\]\[[^\]\n]*\]|"
    r"data:image/|(?:https?://|(?:[A-Za-z]:[\\/]|\.{0,2}[\\/])|\b[\w.-]+[\\/]|\b)"
    r"[^\s<>\"']*\.(?:png|jpe?g|gif|webp|bmp|tiff?|svg)(?=$|[\s?#)<>\"'])|"
    r"\[(?:image|figure|picture)(?:\s+(?:missing|unavailable))?\]|<image>", re.I)
IMAGE_DEPENDENCY = re.compile(
    r"\b(?:shown|displayed|indicated|marked|labelled|labeled|seen)\s+(?:in|on)\s+(?:the\s+)?"
    r"(?:image|figure|picture|diagram|radiograph|x[- ]ray|photograph|chart|micrograph)\b|"
    r"\b(?:image|figure|picture|diagram|radiograph|x[- ]ray|photograph|chart|micrograph)\s+"
    r"(?:shown\s+)?(?:below|above|depicts|shows)\b|"
    r"\b(?:identify|name)\s+the\s+structure\s+(?:shown|indicated)\s+(?:below|above)\b|"
    r"(?:如下图|如图所示|观察下图|根据图中|图中所示|影像所示|识别图中)", re.I)
IMAGE_WORD = re.compile(r"\b(?:image|figure|picture|diagram|radiograph)\b", re.I)
IMAGE_FIELDS = frozenset({"image", "images", "image_url", "image_path", "image_id", "image_payload",
                          "image_resource", "image_attachment", "pixel_values"})
ABBREVIATION = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof|vs|etc|e\.g|i\.e)\.|\b(?:[A-Za-z]\.){2,}|\d\.\d", re.I)
OPTION_LINE = re.compile(r"^\s*(?:[A-D][.)]|\([A-D]\))\s+", re.I)


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", "" if value is None else str(value)).split()).casefold()


def digest(value):
    data = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("ascii")).hexdigest()


def content_from_raw(raw):
    def text(value):
        return "" if value is None else str(value)
    return dict(question=text(raw.get("question")),
                options={l: text(raw.get(f"op{l.lower()}")) for l in LETTERS},
                explanation=text(raw.get("exp")))


def text_fields(content):
    yield "question", content["question"]
    for l in LETTERS:
        yield f"options.{l}", content["options"][l]
    yield "explanation", content["explanation"]


def set_field(content, field, value):
    if field.startswith("options."):
        content["options"][field[-1]] = value
    else:
        content[field] = value


def fingerprints(content, raw_answer):
    status, letter = parse_answer(raw_answer)
    answer = ["valid", letter] if status == "valid" else [status, type(raw_answer).__name__, raw_answer]
    parts = [normalize(content["question"])] + [normalize(content["options"][l]) for l in LETTERS]
    return dict(question=digest(parts[0]), prompt=digest(parts),
                full_record=digest(parts + [answer, normalize(content["explanation"])]))


def image_evidence(raw, content):
    evidence, hints = [], []
    def walk(value, path="original"):
        if isinstance(value, dict):
            # An explicitly typed image attachment need not carry a filename extension.
            if value.get("type") in ("image", "image_url") or str(value.get("mime_type", "")).startswith("image/"):
                evidence.append(dict(field=path, kind="structured_image_attachment", value=copy.deepcopy(value)))
            for k, v in value.items():
                if k.casefold() in IMAGE_FIELDS and v is not False and v not in (None, "", [], {}):
                    evidence.append(dict(field=f"{path}.{k}", kind="structured_image_resource", value=copy.deepcopy(v)))
                walk(v, f"{path}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")
        elif isinstance(value, str):
            for match in IMAGE_RESOURCE.finditer(html.unescape(value)):
                evidence.append(dict(field=path, kind="image_resource", view="entity_decoded",
                                     span=list(match.span()), text=match.group(), original_text=value))
    walk(raw)
    for match in IMAGE_DEPENDENCY.finditer(content["question"]):
        evidence.append(dict(field="question", kind="explicit_image_dependency", span=list(match.span()), text=match.group()))
    if not evidence:
        for field, text in text_fields(content):
            for match in IMAGE_WORD.finditer(text):
                hints.append(dict(code="possible_image_reference", field=field,
                                  span=list(match.span()), text=match.group()))
    return evidence, hints


class FormatHTML(HTMLParser):
    INLINE = frozenset({"b", "strong", "i", "em", "u", "span"})
    BLOCK = frozenset({"p", "div"})

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.unsafe = [], []
        self.stack = []

    def handle_starttag(self, tag, attrs):
        if attrs:
            self.unsafe.append("unpreserved_tag_attributes")
        if tag not in self.INLINE | self.BLOCK | {"br"}:
            self.unsafe.append(tag)
        if tag in self.BLOCK:
            self.parts.append("\n\n")
        elif tag == "br":
            self.parts.append("\n")
        if tag != "br":
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag != "br":
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag not in self.INLINE | self.BLOCK:
            self.unsafe.append(tag)
        if not self.stack or self.stack[-1] != tag:
            self.unsafe.append("unbalanced_markup")
        else:
            self.stack.pop()
        if tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        self.parts.append(data)

    def handle_comment(self, data):
        self.unsafe.append("comment")

    def handle_decl(self, decl):
        self.unsafe.append("declaration")


def clean_html(text):
    decoded = html.unescape(text)
    if html.unescape(decoded) != decoded:
        return text, ["nested_entity_boundary_uncertain"]
    parser = FormatHTML()
    parser.feed(decoded)
    parser.close()
    if parser.stack:
        parser.unsafe.append("unbalanced_markup")
    if parser.unsafe:
        return text, sorted(set(parser.unsafe))
    return "".join(parser.parts), []


def normalize_whitespace(text):
    # Preserve internal alignment, paragraphs and indentation; trim meaningless edges only.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if (re.search(r"(?m)^(?: {4}|\t)\S", text) or
        ("\n" in text and ("```" in text or re.search(r"(?m)^[ \t]+\S", text)))):
        return text.strip("\n")
    return text.strip()


def declarations(text):
    found, offset = [], 0
    for line in text.splitlines(keepends=True):
        match = DECLARATION.match(line.rstrip("\r\n"))
        if match:
            group = 1 if match.group(1) else 2
            found.append(dict(letter=match.group(group).upper(),
                              letter_span=[offset+match.start(group), offset+match.end(group)],
                              span=[offset, offset+len(line.rstrip("\r\n"))],
                              text=line[match.end():].strip(),
                              leading=not text[:offset].strip()))
        offset += len(line)
    return found


def body_without_answers_and_references(text, options, gold):
    """Analysis view only: never removes references or answer lines from stored text."""
    lines = text.splitlines()
    body, uncertain, citation_evidence = [], False, []
    options_normalized = {normalize(v) for v in options.values() if normalize(v)}
    in_reference = False
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            in_reference = False
            i += 1
            continue
        header = REFERENCE_HEADER.fullmatch(line)
        title = REFERENCE_TITLE.fullmatch(line)
        if header or title:
            in_reference = True
            line = header.group(1).strip() if header else ""
            if not line:
                i += 1
                continue
        if in_reference:
            if CITATION.fullmatch(line):
                citation_evidence.append(dict(line=i+1, text=line))
            elif i+1 < len(lines) and EDITION_ONLY.fullmatch(lines[i+1].strip()):
                citation_evidence.append(dict(line=i+1, text=line+"\n"+lines[i+1]))
                i += 1
            else:
                # Keep the material, never assume that all text after Ref is bibliography.
                uncertain = True
                body.append(line)
            i += 1
            continue
        match = DECLARATION.match(line)
        if match:
            tail = line[match.end():].strip()
            if not tail or normalize(tail) in options_normalized:
                i += 1
                continue
            body.append(tail)
        elif re.fullmatch(r"\(?[A-D]\)?[.)]?", line, re.I):
            pass
        elif gold and normalize(line) == normalize(options.get(gold, "")):
            pass
        else:
            body.append(line)
        i += 1
    return "\n".join(body).strip(), uncertain, citation_evidence


def protected_prose(text):
    stripped = text.strip()
    return bool(not stripped or stripped[0] in "#>`\"'" or "```" in text or
                re.search(r"[=<>]|\$|\\\(|\\\[", text) or stripped.isupper() or
                any(OPTION_LINE.match(line) or REFERENCE_HEADER.match(line) or DECLARATION.match(line)
                    for line in text.splitlines()))


def complete_prose(text):
    return bool(not protected_prose(text) and re.search(r"[.!?]$", text.strip()) and
                len(text.split()) > 1 and not ABBREVIATION.search(text))


def remove_consecutive_duplicates(text):
    if "```" in text or re.search(r"(?m)^[ \t]+\S", text):
        return text, []
    edits = []
    # Paragraphs have explicit blank-line boundaries. Keep source spans in the current text.
    chunks = list(re.finditer(r"\S(?:.*?\S)?(?=\n[ \t]*\n|$)", text, re.S))
    previous = None
    removals = []
    for chunk in chunks:
        if previous is not None and complete_prose(chunk.group()) and complete_prose(previous.group()) and normalize(chunk.group()) == normalize(previous.group()):
            removals.append((chunk.start(), chunk.end(), chunk.group(), "consecutive_exact_duplicate_paragraph_removed"))
        else:
            previous = chunk
    for start, end, fragment, rule in reversed(removals):
        before = text
        text = text[:start] + text[end:]
        edits.append(dict(rule=rule, before=before, after=text, span=[start, end], removed=fragment))
    # Sentence deletion only where every character has an unambiguous sentence/space parse.
    # Any abbreviation/decimal or protected block disables sentence edits in that paragraph.
    offset = 0
    sentences_to_remove = []
    for para in re.split(r"(\n[ \t]*\n)", text):
        if not complete_prose(para):
            offset += len(para)
            continue
        matches = list(re.finditer(r"[^.!?\n]+[.!?](?:[ \t]+|$)", para))
        if "".join(m.group() for m in matches) != para:
            offset += len(para)
            continue
        previous = None
        for match in matches:
            sentence = match.group().strip()
            if not complete_prose(sentence):
                previous = None
            elif previous is not None and normalize(previous) == normalize(sentence):
                sentences_to_remove.append((offset+match.start(), offset+match.end(), match.group()))
            else:
                previous = sentence
        offset += len(para)
    for start, end, fragment in reversed(sentences_to_remove):
        before = text
        text = text[:start] + text[end:]
        edits.append(dict(rule="consecutive_exact_duplicate_sentence_removed", before=before,
                          after=text, span=[start, end], removed=fragment))
    return text, edits


def classify_record(raw, input_index=0, source=None):
    original = copy.deepcopy(raw)
    content = content_from_raw(raw)
    work = copy.deepcopy(content)
    status, gold = parse_answer(raw.get("cop"))
    reasons, evidence, hints, proposed = [], [], [], []
    d_reasons, c_reasons = [], []
    images, image_hints = image_evidence(raw, content)
    evidence.extend(images)
    hints.extend(image_hints)
    if images:
        d_reasons.append("contains_image_or_image_dependency")
    if not content["question"].strip():
        d_reasons.append("missing_question")
    if status == "missing":
        d_reasons.append("missing_gold_answer")
    elif status == "invalid":
        c_reasons.append("invalid_gold_answer")
    if not content["explanation"].strip():
        d_reasons.append("missing_explanation")
    elif content["explanation"].strip().casefold() in PLACEHOLDERS:
        d_reasons.append("placeholder_explanation")

    def edit(field, after, rule, **details):
        before = dict(text_fields(work))[field]
        if before != after:
            details.setdefault("span", [0, len(before)])
            proposed.append(dict(field=field, before=before, after=after, rule=rule,
                                 explanation="Frozen local safe repair; original gold and source bytes unchanged", **details))
            set_field(work, field, after)

    if not images:
        for field, text in list(text_fields(work)):
            cleaned, unsafe = clean_html(text)
            if unsafe:
                c_reasons.append("html_structure_unsafe")
                evidence.append(dict(field=field, kind="unsafe_html", tags=unsafe))
            else:
                edit(field, cleaned, "html_format_safe_cleanup")
            edit(field, normalize_whitespace(dict(text_fields(work))[field]), "whitespace_safe_cleanup")
    # Recheck decoded visible fields; empty formatting cannot hide missing content.
    if not work["question"].strip() and "missing_question" not in d_reasons:
        d_reasons.append("missing_question")
    opts = work["options"]
    empty = [not opts[l].strip() for l in LETTERS]
    labels = [opts[l].strip().upper() in (l, f"{l}.", f"{l})", f"({l})", f"[{l}]") for l in LETTERS]
    if all(a or b for a, b in zip(empty, labels)):
        d_reasons.append("all_options_missing_or_placeholder")
    elif any(empty):
        c_reasons.append("missing_option")
    nonempty_options = [normalize(opts[l]) for l in LETTERS if normalize(opts[l])]
    if len(nonempty_options) != len(set(nonempty_options)):
        c_reasons.append("duplicate_option_exact")
    if not work["explanation"].strip():
        d_reasons.append("missing_explanation")
    elif work["explanation"].strip().casefold() in PLACEHOLDERS:
        d_reasons.append("placeholder_explanation")

    for field, text in text_fields(work):
        damaged = list(re.finditer(r"[\ufffd\ufffe\uffff\ud800-\udfff]", text))
        if damaged:
            c_reasons.append("text_corruption_unrepairable")
            evidence.extend(dict(field=field, kind="damaged_unicode", span=list(m.span()), text=m.group()) for m in damaged)
        for match in re.finditer(r"\b(\w+)\s+\1\b|\b[a-z]{6,}[A-Z][a-z]{4,}\b|Ã.|Â.|â€", text):
            hints.append(dict(code="possible_text_artifact", field=field, span=list(match.span()), text=match.group()))

    claims = declarations(work["explanation"])
    evidence.extend(dict(kind="answer_declaration", **claim) for claim in claims)
    if len({c["letter"] for c in claims}) > 1:
        c_reasons.append("contradictory_answer_declarations")
    elif status == "valid":
        for claim in claims:
            if claim["letter"] == gold:
                continue
            matching = [l for l in LETTERS if normalize(claim["text"]) and normalize(claim["text"]) == normalize(opts[l])]
            if matching == [gold] and claim["leading"]:
                start, end = claim["letter_span"]
                replacement = work["explanation"][:start] + gold + work["explanation"][end:]
                edit("explanation", replacement, "answer_letter_typo_safe_repair", span=[start, end],
                     old_letter=claim["letter"], new_letter=gold, matched_option=gold)
            elif len(matching) == 1 and matching[0] != gold:
                c_reasons.append("explicit_answer_content_conflict")
            else:
                c_reasons.append("answer_text_mapping_uncertain")
    body, uncertain_ref, citations = body_without_answers_and_references(work["explanation"], opts, gold)
    evidence.extend(dict(kind="reference", **citation) for citation in citations)
    if uncertain_ref:
        c_reasons.append("reference_body_boundary_uncertain")
    elif not body and "html_structure_unsafe" not in c_reasons:
        d_reasons.append("answer_only_no_explanation_body")

    # Never delete material in an unsafe structure or uncertain citation block.
    if not d_reasons and not c_reasons:
        cleaned, repeated_edits = remove_consecutive_duplicates(work["explanation"])
        for item in repeated_edits:
            before = item.pop("before")
            after = item.pop("after")
            edit("explanation", after, **item)
        edit("explanation", normalize_whitespace(cleaned), "whitespace_safe_cleanup")

    quality = "D" if d_reasons else "C" if c_reasons else "B" if proposed else "A"
    repairs = proposed if quality == "B" else []
    effective = work if quality == "B" else content
    reasons = list(dict.fromkeys(d_reasons + c_reasons + ([p["rule"] for p in repairs])))
    if not reasons:
        reasons = ["no_explicit_issue_found"]
    for event, change in enumerate(repairs, 1):
        change.update(id=raw.get("id"), input_index=input_index, event_index=event)
    return dict(id=raw.get("id"), input_index=input_index, source=source or {}, original=original,
                content=effective, gold_answer=gold, answer_parse_status=status, quality_class=quality,
                reason_codes=reasons, quality_block_reasons=list(dict.fromkeys(d_reasons+c_reasons)),
                evidence=evidence, weak_hints=hints, repairs=repairs,
                uncommitted_repair_proposals=proposed if quality in ("C", "D") else [],
                fingerprints_original=fingerprints(content, raw.get("cop")),
                fingerprints_repaired=fingerprints(effective, raw.get("cop")),
                length_status="not_checked", duplicate_flags=[], duplicate_status="unique",
                training_status=None, training_block_reasons=[], rules_version=VERSION)


def quality_conflict(record, reason):
    if record["quality_class"] != "D":
        record["quality_class"] = "C"
    record["reason_codes"] = [code for code in record["reason_codes"] if code != "no_explicit_issue_found"]
    if reason not in record["reason_codes"]:
        record["reason_codes"].append(reason)
    if reason not in record["quality_block_reasons"]:
        record["quality_block_reasons"].append(reason)


def assign_training_status(record):
    blocks = []
    q = record["quality_class"]
    if q in ("C", "D"):
        blocks.append(f"quality_{q}")
        blocks.extend(record["quality_block_reasons"])
    blocks.extend(flag for flag in record["duplicate_flags"] if flag != "same_prompt_multiple_explanations")
    if record["length_status"] == "over_limit":
        blocks.append("over_512")
    if q in ("A", "B") and record["length_status"] == "not_checked":
        raise ValueError("A/B cannot become eligible before the real length check")
    record["training_block_reasons"] = list(dict.fromkeys(blocks))
    record["training_status"] = (
        "excluded_quality_D" if q == "D" else "isolated_quality_C" if q == "C" else
        "isolated_dev_overlap" if "dev_eval_overlap" in blocks else
        "isolated_duplicate" if "exact_duplicate_copy" in blocks else
        "isolated_over_512" if "over_512" in blocks else "eligible_candidate")
    record["duplicate_status"] = next((f for f in (
        "dev_eval_overlap", "duplicate_answer_conflict", "exact_duplicate_copy", "same_prompt_multiple_explanations"
    ) if f in record["duplicate_flags"]), "unique")


def apply_dataset_checks(records, dev_records):
    """Operate on effective content; first representative follows supplied input order."""
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate input ID: cannot silently overwrite a record")
    groups, questions, full_groups, dev_by_prompt = (defaultdict(list) for _ in range(4))
    for dev in dev_records:
        if parse_answer(dev.get("cop"))[0] != "valid":
            raise ValueError("Invalid dev gold; cannot safely check cross-split conflicts")
        fp = fingerprints(content_from_raw(dev), dev.get("cop"))
        dev_by_prompt[fp["prompt"]].append(dev)
    for r in records:
        fp = r["fingerprints_repaired"]
        groups[fp["prompt"]].append(r)
        questions[fp["question"]].append(r)
        full_groups[fp["full_record"]].append(r)
    reports = []
    def flag(r, value):
        if value not in r["duplicate_flags"]:
            r["duplicate_flags"].append(value)
    for kind, mapping in (("question_only", questions), ("prompt", groups), ("full_record", full_groups)):
        for key, members in mapping.items():
            if len(members) < 2:
                continue
            report = dict(kind=kind, fingerprint=key, ids=[r["id"] for r in members],
                          input_indices=[r["input_index"] for r in members],
                          gold_answers=[r["gold_answer"] for r in members],
                          explanation_fingerprints=[digest(normalize(r["content"]["explanation"])) for r in members])
            if kind == "full_record":
                report["representative_id"] = members[0]["id"]
                for r in members[1:]:
                    flag(r, "exact_duplicate_copy")
                    r["duplicate_representative_id"] = members[0]["id"]
            if kind == "prompt":
                valid = {r["gold_answer"] for r in members if r["answer_parse_status"] == "valid"}
                if len(valid) > 1:
                    for r in members:
                        flag(r, "duplicate_answer_conflict")
                        quality_conflict(r, "duplicate_answer_conflict")
                    report["relation"] = "duplicate_answer_conflict"
                elif len(valid) == 1 and all(r["answer_parse_status"] == "valid" for r in members) and len({normalize(r["content"]["explanation"]) for r in members}) > 1:
                    for r in members:
                        flag(r, "same_prompt_multiple_explanations")
                    report["relation"] = "same_prompt_multiple_explanations"
            reports.append(report)
    for r in records:
        matches = dev_by_prompt.get(r["fingerprints_repaired"]["prompt"], [])
        if matches:
            flag(r, "dev_eval_overlap")
            r["dev_overlap_ids"] = [d["id"] for d in matches]
            if r["answer_parse_status"] == "valid" and any(parse_answer(d["cop"])[1] != r["gold_answer"] for d in matches):
                flag(r, "dev_answer_conflict")
                quality_conflict(r, "dev_answer_conflict")
        assign_training_status(r)
    return reports


def summarize(records, groups):
    classes = {c: sum(r["quality_class"] == c for r in records) for c in LETTERS}
    blocks = Counter(b for r in records for b in r["training_block_reasons"])
    ab = [r for r in records if r["quality_class"] in ("A", "B")]
    long = [r for r in ab if r["length_status"] == "over_limit"]
    lengths = sorted(r["full_token_length"] for r in long)
    def fraction(n, d):
        return n/d if d else 0.0
    return dict(input_records=len(records), quality_counts=classes,
                quality_partition_passed=sum(classes.values()) == len(records),
                reason_counts=dict(sorted(Counter(c for r in records for c in r["reason_codes"]).items())),
                training_status_counts=dict(sorted(Counter(r["training_status"] for r in records).items())),
                training_block_reason_counts=dict(sorted(blocks.items())),
                blocked_records_union=sum(bool(r["training_block_reasons"]) for r in records),
                blocked_ids_union=sorted(r["id"] for r in records if r["training_block_reasons"]),
                length=dict(final_AB_checked=len(ab), within_limit=len(ab)-len(long), over_limit=len(long),
                            within_fraction=fraction(len(ab)-len(long), len(ab)), over_fraction=fraction(len(long), len(ab)),
                            over_limit_by_quality=dict(Counter(r["quality_class"] for r in long)),
                            over_limit_distribution=dict(min=lengths[0], max=lengths[-1],
                                median=lengths[len(lengths)//2], p95=lengths[int((len(lengths)-1)*.95)]) if lengths else {},
                            all_measured_including_later_conflicts=sum(r["length_status"] != "not_checked" for r in records)),
                explanation_char_statistics=dict(under_50=sum(len(r["content"]["explanation"]) < 50 for r in records),
                                                  over_1000=sum(len(r["content"]["explanation"]) > 1000 for r in records)),
                duplicate_group_counts=dict(Counter(g["kind"] for g in groups)),
                same_prompt_multiple_explanations_groups=sum(g.get("relation") == "same_prompt_multiple_explanations" for g in groups),
                repair_events=sum(len(r["repairs"]) for r in records), training_release_ready=False)
