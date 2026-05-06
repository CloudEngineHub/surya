"""Surya2 layout labels emitted by the model + canonicalization to surya's
public label vocabulary."""

# Labels the model emits via LAYOUT_PROMPT (see surya/inference/prompts.py).
LAYOUT_LABELS = (
    "Caption",
    "Footnote",
    "Equation-Block",
    "List-Group",
    "Page-Header",
    "Page-Footer",
    "Image",
    "Section-Header",
    "Table",
    "Text",
    "Complex-Block",
    "Code-Block",
    "Form",
    "Table-Of-Contents",
    "Figure",
    "Chemical-Block",
    "Diagram",
    "Bibliography",
    "Blank-Page",
)

# Canonicalize raw model labels to public surya label names. Marker and other
# downstream consumers depend on these names.
LAYOUT_PRED_RELABEL = {
    "Caption": "Caption",
    "Footnote": "Footnote",
    "Equation-Block": "Equation",
    "List-Group": "ListGroup",
    "Page-Header": "PageHeader",
    "Page-Footer": "PageFooter",
    "Image": "Picture",
    "Section-Header": "SectionHeader",
    "Table": "Table",
    "Text": "Text",
    "Complex-Block": "Figure",
    "Code-Block": "Code",
    "Form": "Form",
    "Table-Of-Contents": "TableOfContents",
    "Figure": "Figure",
    "Chemical-Block": "ChemicalBlock",
    "Diagram": "Diagram",
    "Bibliography": "Bibliography",
    "Blank-Page": "BlankPage",
}
