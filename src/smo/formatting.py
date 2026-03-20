from __future__ import annotations

import re
from html import escape


def build_visit_summary_html(
    *,
    patient_id: str,
    timestamp: str,
    age: float,
    weight: float,
    symptoms: str,
    spo2: float,
    heart_rate: float,
    history: str,
) -> str:
    return (
        "<div class='markdown-body'>"
        "<div class='summary-grid'>"
        "<div class='summary-item'>"
        "<span class='summary-label'>Patient ID</span>"
        f"<span class='summary-value'><code>{escape(patient_id)}</code></span>"
        "</div>"
        "<div class='summary-item'>"
        "<span class='summary-label'>Timestamp</span>"
        f"<span class='summary-value'>{escape(timestamp)}</span>"
        "</div>"
        "<div class='summary-item'>"
        "<span class='summary-label'>Age</span>"
        f"<span class='summary-value'>{escape(str(age))}</span>"
        "</div>"
        "<div class='summary-item'>"
        "<span class='summary-label'>Weight</span>"
        f"<span class='summary-value'>{escape(str(weight))} kg</span>"
        "</div>"
        "<div class='summary-item summary-item-wide'>"
        "<span class='summary-label'>Symptoms</span>"
        f"<span class='summary-value'>{escape(symptoms)}</span>"
        "</div>"
        "<div class='summary-item'>"
        "<span class='summary-label'>SpO2 Level</span>"
        f"<span class='summary-value'>{escape(str(spo2))}%</span>"
        "</div>"
        "<div class='summary-item'>"
        "<span class='summary-label'>Heart Rate</span>"
        f"<span class='summary-value'>{escape(str(heart_rate))} bpm</span>"
        "</div>"
        "<div class='summary-item summary-item-wide'>"
        "<span class='summary-label'>Medical History</span>"
        f"<span class='summary-value'>{escape(history)}</span>"
        "</div>"
        "</div>"
        "</div>"
    )


def _format_inline(text: str) -> str:
    parts: list[str] = []
    cursor = 0
    for match in re.finditer(r"\*\*(.+?)\*\*", text):
        parts.append(escape(text[cursor : match.start()]))
        parts.append(f"<strong>{escape(match.group(1))}</strong>")
        cursor = match.end()
    parts.append(escape(text[cursor:]))
    return "".join(parts)


def _render_text_block(text: str) -> str:
    html_parts = ["<div class='markdown-body formatted-output'>"]
    in_list = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            continue

        numbered_match = re.match(r"^(\d+)\.\s+(.*)$", line)
        bullet_match = re.match(r"^-\s+(.*)$", line)

        if numbered_match:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(
                "<p class='opinion-step'>"
                f"<span class='step-index'>{escape(numbered_match.group(1))}.</span> "
                f"{_format_inline(numbered_match.group(2))}"
                "</p>"
            )
            continue

        if bullet_match:
            if not in_list:
                html_parts.append("<ul class='opinion-list'>")
                in_list = True
            html_parts.append(f"<li>{_format_inline(bullet_match.group(1))}</li>")
            continue

        if in_list:
            html_parts.append("</ul>")
            in_list = False
        html_parts.append(f"<p>{_format_inline(line)}</p>")

    if in_list:
        html_parts.append("</ul>")

    html_parts.append("</div>")
    return "".join(html_parts)


def _split_numbered_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    active_key: str | None = None
    active_lines: list[str] = []

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        match = re.match(r"^([1-4])\.\s+(.*)$", stripped)
        if match:
            if active_key is not None:
                sections[active_key] = "\n".join(active_lines).strip()
            active_key = match.group(1)
            active_lines = [match.group(2).strip()]
            continue

        if active_key is not None:
            active_lines.append(stripped)

    if active_key is not None:
        sections[active_key] = "\n".join(active_lines).strip()

    return sections


def _strip_markdown_heading(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("**") and stripped.endswith("**"):
        stripped = stripped[2:-2].strip()
    return stripped.rstrip(":").strip()


def _extract_treatment_heading(content: str) -> tuple[str | None, str | None]:
    stripped = content.strip()

    bold_match = re.match(r"^\*\*(.+?)\*\*(?:\s*[:\-]?\s*(.*))?$", stripped)
    if bold_match:
        title = _strip_markdown_heading(f"**{bold_match.group(1)}**")
        remainder = (bold_match.group(2) or "").strip()
        return title, remainder or None

    plain_match = re.match(r"^([A-Za-z][A-Za-z0-9 /,&()\-]{1,48}):\s*(.*)$", stripped)
    if plain_match:
        title = plain_match.group(1).strip()
        remainder = plain_match.group(2).strip()
        return title, remainder or None

    if len(stripped) <= 48 and stripped.endswith(":"):
        return stripped[:-1].strip(), None

    return None, None


def _parse_treatment_plan(text: str) -> tuple[list[str], list[dict[str, object]]]:
    intro_lines: list[str] = []
    groups: list[dict[str, object]] = []
    active_group: dict[str, object] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        bullet_match = re.match(r"^-\s+(.*)$", line)
        content = bullet_match.group(1).strip() if bullet_match else line
        heading, inline_item = _extract_treatment_heading(content)

        if heading:
            active_group = {"title": heading, "items": []}
            groups.append(active_group)
            if inline_item:
                active_group["items"].append(inline_item)
            continue

        if active_group is not None:
            active_group["items"].append(content)
        else:
            intro_lines.append(content)

    return intro_lines, groups


def _build_treatment_plan_html(text: str) -> str:
    intro_lines, groups = _parse_treatment_plan(text)
    if not groups:
        return _render_text_block(text)

    html_parts = ["<div class='markdown-body treatment-output'>"]
    for line in intro_lines:
        html_parts.append(f"<p>{_format_inline(line)}</p>")

    html_parts.append("<div class='treatment-groups'>")
    for group in groups:
        html_parts.append("<section class='treatment-group'>")
        html_parts.append(f"<h4>{escape(str(group['title']))}</h4>")
        items = [str(item) for item in group["items"]]
        if items:
            html_parts.append("<ul class='treatment-items'>")
            for item in items:
                html_parts.append(f"<li>{_format_inline(item)}</li>")
            html_parts.append("</ul>")
        html_parts.append("</section>")
    html_parts.append("</div></div>")
    return "".join(html_parts)


def _estimate_assessment_units(section_text: str) -> int:
    normalized = re.sub(r"\s+", " ", section_text.strip())
    return 78 + len(normalized) // 4


def _estimate_treatment_units(group: dict[str, object]) -> int:
    title = str(group["title"])
    items = [str(item) for item in group["items"]]
    normalized = " ".join([title, *items])
    return 54 + len(normalized) // 4 + max(0, len(items) - 1) * 10


def _estimate_treatment_item_units(item: str) -> int:
    normalized = re.sub(r"\s+", " ", item.strip())
    return 18 + len(normalized) // 4


def _split_treatment_group(group: dict[str, object], *, limit: int = 170) -> list[dict[str, object]]:
    title = str(group["title"])
    items = [str(item) for item in group["items"]]
    if not items:
        return [group]

    parts: list[dict[str, object]] = []
    current_items: list[str] = []
    current_units = 34 + len(title) // 4

    for item in items:
        item_units = _estimate_treatment_item_units(item)
        if current_items and current_units + item_units > limit:
            parts.append(
                {
                    "title": title if not parts else f"{title} (cont.)",
                    "items": list(current_items),
                }
            )
            current_items = []
            current_units = 34 + len(title) // 4
        current_items.append(item)
        current_units += item_units

    if current_items:
        parts.append(
            {
                "title": title if not parts else f"{title} (cont.)",
                "items": list(current_items),
            }
        )
    return parts


def _build_paper_card_fragment(
    *,
    label: str,
    body_html: str,
    tools_used: str | None = None,
) -> str:
    parts = ["<section class='output-card opinion-card paper-card'>"]
    parts.append(f"<div class='pill-label'>{escape(label)}</div>")
    if tools_used:
        parts.append(
            "<p class='meta-row'>"
            f"<span><strong>Tools used:</strong> {escape(tools_used)}</span>"
            "</p>"
        )
    parts.append(f"<div class='rich-output'>{body_html}</div>")
    parts.append("</section>")
    return "".join(parts)


def _build_assessment_cards(
    sections: dict[str, str],
    selected_tools: list[str] | None,
) -> str:
    entries = [(idx, sections[idx]) for idx in ("1", "2", "3") if sections.get(idx)]
    if not entries:
        return ""

    cards: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_units = 34 if selected_tools else 0
    limit = 300

    for idx, content in entries:
        section_units = _estimate_assessment_units(content)
        if current and current_units + section_units > limit:
            cards.append(current)
            current = []
            current_units = 0
        current.append((idx, content))
        current_units += section_units

    if current:
        cards.append(current)

    tool_labels = ", ".join(selected_tools or [])
    fragments: list[str] = []
    for card_index, card_sections in enumerate(cards):
        card_text = "\n\n".join(f"{idx}. {content}" for idx, content in card_sections)
        fragments.append(
            _build_paper_card_fragment(
                label="Clinical Assessment" if card_index == 0 else "Clinical Assessment (cont.)",
                tools_used=tool_labels if card_index == 0 and tool_labels else None,
                body_html=_render_text_block(card_text),
            )
        )
    return "".join(fragments)


def _build_treatment_card_body(
    intro_lines: list[str],
    groups: list[dict[str, object]],
) -> str:
    if not groups and intro_lines:
        return _render_text_block("\n".join(intro_lines))

    html_parts = ["<div class='markdown-body treatment-output'>"]
    for line in intro_lines:
        html_parts.append(f"<p>{_format_inline(line)}</p>")
    html_parts.append("<div class='treatment-groups'>")
    for group in groups:
        html_parts.append("<section class='treatment-group'>")
        html_parts.append(f"<h4>{escape(str(group['title']))}</h4>")
        items = [str(item) for item in group["items"]]
        if items:
            html_parts.append("<ul class='treatment-items'>")
            for item in items:
                html_parts.append(f"<li>{_format_inline(item)}</li>")
            html_parts.append("</ul>")
        html_parts.append("</section>")
    html_parts.append("</div></div>")
    return "".join(html_parts)


def _build_treatment_cards(text: str) -> str:
    intro_lines, groups = _parse_treatment_plan(text)
    if not groups:
        return _build_paper_card_fragment(
            label="Treatment Plan",
            body_html=_render_text_block(text),
        )

    expanded_groups: list[dict[str, object]] = []
    for group in groups:
        expanded_groups.extend(_split_treatment_group(group))

    cards: list[tuple[list[str], list[dict[str, object]]]] = []
    current_groups: list[dict[str, object]] = []
    current_units = 0
    limit = 300
    intro_units = sum(24 + len(re.sub(r"\s+", " ", line.strip())) // 5 for line in intro_lines)

    for group in expanded_groups:
        group_units = _estimate_treatment_units(group)
        added_intro_units = intro_units if not cards and not current_groups else 0
        if current_groups and current_units + group_units + added_intro_units > limit:
            cards.append((intro_lines if not cards else [], current_groups))
            current_groups = []
            current_units = 0
            added_intro_units = 0
        current_groups.append(group)
        current_units += group_units + added_intro_units

    if current_groups or intro_lines:
        cards.append((intro_lines if not cards else [], current_groups))

    fragments: list[str] = []
    for card_index, (card_intro, card_groups) in enumerate(cards):
        fragments.append(
            _build_paper_card_fragment(
                label="Treatment Plan" if card_index == 0 else "Treatment Plan (cont.)",
                body_html=_build_treatment_card_body(card_intro, card_groups),
            )
        )
    return "".join(fragments)


def build_opinion_panels_html(
    text: str,
    *,
    selected_tools: list[str] | None = None,
) -> tuple[str, str]:
    sections = _split_numbered_sections(text)
    if not sections:
        fallback = _build_paper_card_fragment(
            label="Clinical Assessment",
            tools_used=", ".join(selected_tools or []) or None,
            body_html=_render_text_block(text),
        )
        return fallback, ""

    treatment_text = sections.get("4", "").strip()
    treatment_text = re.sub(
        r"^(recommended|updated)?\s*treatment plan(?: adjustments)?:\s*",
        "",
        treatment_text,
        flags=re.IGNORECASE,
    )

    assessment_html = _build_assessment_cards(sections, selected_tools)
    treatment_html = _build_treatment_cards(treatment_text or sections.get("4", ""))
    return assessment_html, treatment_html


def plain_text_to_html(text: str) -> str:
    return _render_text_block(text)
