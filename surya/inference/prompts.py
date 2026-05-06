"""Prompt strings for surya2. The exact wording is the model's training-time
contract — do not paraphrase without retraining."""

from surya.inference.schema import PROMPT_TYPE_BLOCK as PROMPT_TYPE_BLOCK
from surya.inference.schema import (
    PROMPT_TYPE_HIGH_ACCURACY_BBOX as PROMPT_TYPE_HIGH_ACCURACY_BBOX,
)
from surya.inference.schema import PROMPT_TYPE_LAYOUT as PROMPT_TYPE_LAYOUT
from surya.inference.schema import PROMPT_TYPE_TABLE_REC as PROMPT_TYPE_TABLE_REC

ALLOWED_TAGS = [
    "math",
    "br",
    "i",
    "b",
    "u",
    "del",
    "sup",
    "sub",
    "table",
    "tr",
    "td",
    "p",
    "th",
    "div",
    "pre",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "ul",
    "ol",
    "li",
    "input",
    "a",
    "span",
    "img",
    "hr",
    "tbody",
    "small",
    "caption",
    "strong",
    "thead",
    "big",
    "code",
    "chem",
]

ALLOWED_ATTRIBUTES = [
    "class",
    "colspan",
    "rowspan",
    "display",
    "checked",
    "type",
    "border",
    "value",
    "style",
    "href",
    "alt",
    "align",
    "data-bbox",
    "data-label",
]

# Block labels we don't run OCR on.
SKIP_OCR_LABELS = {"Figure", "Image", "Diagram", "Blank-Page"}

LAYOUT_LABELS = """- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure
- Chemical-Block
- Diagram
- Bibliography
- Blank-Page"""

GUIDELINES = f"""Only use these tags {ALLOWED_TAGS}, and these attributes {ALLOWED_ATTRIBUTES}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural."""

LAYOUT_PROMPT = (
    "Output the layout of this image as JSON. Each entry is a dict with "
    '"label", "bbox", and "count" fields. Bbox is x0 y0 x1 y1, normalized 0-1000.'
)

BLOCK_PROMPT = "OCR this block image to HTML."

TABLE_REC_PROMPT = (
    "Output the table rows then columns as JSON. Each entry is a dict with "
    '"label" ("Row" or "Col") and "bbox" (x0 y0 x1 y1, normalized 0-1000).'
)

HIGH_ACCURACY_BBOX_PROMPT = (
    "OCR this image to HTML. Each block is a div with data-label and data-bbox "
    "(x0 y0 x1 y1, normalized 0-1000)."
)


PROMPT_MAPPING = {
    "layout": LAYOUT_PROMPT,
    "block": BLOCK_PROMPT,
    "table_rec": TABLE_REC_PROMPT,
    "high_accuracy_bbox": HIGH_ACCURACY_BBOX_PROMPT,
}


# JSON schema for LAYOUT_PROMPT — enforced via vllm guided decoding so the
# model can't emit malformed JSON. bbox is a "x0 y0 x1 y1" string (model's
# training-time format); count is a non-negative integer.
LAYOUT_LABEL_SET = [
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
]

LAYOUT_JSON_SCHEMA = {
    "type": "array",
    "maxItems": 200,
    "items": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": LAYOUT_LABEL_SET},
            "bbox": {
                "type": "string",
                "pattern": r"^\d{1,4} \d{1,4} \d{1,4} \d{1,4}$",
            },
            "count": {"type": "integer", "minimum": 0, "maximum": 10000},
        },
        "required": ["label", "bbox", "count"],
        "additionalProperties": False,
    },
}


# JSON schema for TABLE_REC_PROMPT — array of {label: Row|Col, bbox: "x0 y0 x1 y1"}.
TABLE_REC_LABEL_SET = ["Row", "Col"]

TABLE_REC_JSON_SCHEMA = {
    "type": "array",
    "maxItems": 200,
    "items": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": TABLE_REC_LABEL_SET},
            "bbox": {
                "type": "string",
                "pattern": r"^\d{1,4} \d{1,4} \d{1,4} \d{1,4}$",
            },
        },
        "required": ["label", "bbox"],
        "additionalProperties": False,
    },
}
