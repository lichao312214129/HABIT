"""Generate HTML cards for habitat template selection (Step 1 of wizard)."""
from typing import Any, Dict, List

from habit.gui.template_loader import list_habitat_templates


def _card_icon(template_id: str) -> str:
    """Pick a simple emoji icon per template id (cosmetic only)."""
    icons = {
        "liver_dce_two_step": "Liver",
        "mpmri_two_step": "MRI",
        "tumor_quick_one_step": "Quick",
        "tumor_radiomics_two_step": "Radiomics",
        "research_custom": "Custom",
    }
    return icons.get(template_id, "Template")


def render_template_cards_html(current_template_id: str = "") -> str:
    """Render all habitat templates as clickable cards (HTML for gr.HTML).

    Each card shows: icon, display_name, description, expected_modalities,
    clustering_mode, prep_preset.
    """
    templates: List[Dict[str, Any]] = list_habitat_templates()
    if not templates:
        return "<p>No templates found.</p>"

    cards: List[str] = []
    for doc in templates:
        tid = str(doc.get("id", ""))
        name = str(doc.get("display_name", tid))
        desc = str(doc.get("description", ""))
        modalities = doc.get("expected_modalities", [])
        clustering = str(doc.get("clustering_mode", "two_step"))
        prep = str(doc.get("prep_preset", "standard"))
        icon = _card_icon(tid)
        is_active = (tid == current_template_id)
        active_cls = "habit-tpl-active" if is_active else ""

        mod_html = ""
        if modalities:
            mod_chips = "".join(
                f"<span class='habit-tpl-chip'>{m}</span>" for m in modalities
            )
            mod_html = f"<div class='habit-tpl-mods'>{mod_chips}</div>"

        cards.append(
            f"<div class='habit-tpl-card {active_cls}' data-tpl='{tid}'>"
            f"<div class='habit-tpl-icon'>{icon}</div>"
            f"<div class='habit-tpl-body'>"
            f"<div class='habit-tpl-name'>{name}</div>"
            f"<div class='habit-tpl-desc'>{desc}</div>"
            f"{mod_html}"
            f"<div class='habit-tpl-meta'>"
            f"<span>Clustering: {clustering}</span>"
            f"<span>Preprocessing: {prep}</span>"
            f"</div>"
            f"</div></div>"
        )

    return (
        f"<div class='habit-tpl-grid'>" + "".join(cards) + "</div>"
            + "<style>"
            + ".habit-tpl-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px;margin:10px 0;}"
            + ".habit-tpl-card{display:flex;gap:12px;padding:14px;border:1px solid #ddd;border-radius:10px;cursor:pointer;transition:all .15s;background:var(--background-fill-primary,#fff);}"
            + ".habit-tpl-card:hover{border-color:#2563EB;box-shadow:0 2px 8px rgba(37,99,235,0.15);}"
            + ".habit-tpl-card.habit-tpl-active{border-color:#2563EB;background:rgba(37,99,235,0.06);box-shadow:0 2px 8px rgba(37,99,235,0.2);}"
            + ".habit-tpl-icon{font-size:1.4rem;font-weight:700;color:#2563EB;min-width:40px;display:flex;align-items:center;justify-content:center;background:rgba(37,99,235,0.1);border-radius:8px;padding:4px 8px;height:fit-content;}"
            + ".habit-tpl-body{flex:1;min-width:0;}"
            + ".habit-tpl-name{font-weight:600;font-size:0.95rem;margin-bottom:4px;color:var(--body-text-color,#333);}"
            + ".habit-tpl-desc{font-size:0.8rem;color:#6B7280;line-height:1.4;margin-bottom:6px;}"
            + ".habit-tpl-mods{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:6px;}"
            + ".habit-tpl-chip{font-size:0.7rem;background:rgba(0,0,0,0.06);padding:2px 8px;border-radius:10px;color:#4B5563;}"
            + ".habit-tpl-meta{display:flex;gap:12px;font-size:0.7rem;color:#9CA3AF;}"
            + ".habit-tpl-meta span{white-space:nowrap;}"
            + "</style>"
    )


# Click template cards to select the matching gr.Radio option (habitat wizard Step 1).
HABIT_TEMPLATE_CARD_JS = """() => {
    function habitSelectTemplateRadio(templateId) {
        if (!templateId) {
            return;
        }
        const radios = document.querySelectorAll('input[type="radio"]');
        for (const input of radios) {
            if (input.value === templateId) {
                input.click();
                input.dispatchEvent(new Event("input", { bubbles: true }));
                input.dispatchEvent(new Event("change", { bubbles: true }));
                break;
            }
        }
    }
    function habitBindTemplateCards() {
        document.querySelectorAll(".habit-tpl-card[data-tpl]").forEach((card) => {
            if (card.dataset.habitBound === "1") {
                return;
            }
            card.dataset.habitBound = "1";
            card.addEventListener("click", () => {
                habitSelectTemplateRadio(card.getAttribute("data-tpl"));
            });
        });
    }
    habitBindTemplateCards();
    if (!window.__habitTemplateCards) {
        window.__habitTemplateCards = true;
        new MutationObserver(habitBindTemplateCards).observe(
            document.documentElement,
            { childList: true, subtree: true }
        );
    }
}"""


__all__ = ["render_template_cards_html", "HABIT_TEMPLATE_CARD_JS"]
