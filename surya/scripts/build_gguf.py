"""Build a GGUF artifact from a surya VLM checkpoint, suitable for stock llama.cpp.

Three patches are applied to llama.cpp's convert_hf_to_gguf.py before
running the HF→GGUF conversion. All fixes are baked into the output
artifact, so the resulting GGUF runs on unpatched llama.cpp.

Two checkpoint-side patches are also applied to a working copy:
  - config.json:           architectures → ["Qwen3_5ForConditionalGeneration"]
  - tokenizer_config.json: tokenizer_class → "PreTrainedTokenizerFast",
                           strip backend / extra_special_tokens / is_local

The original checkpoint is never modified — patches land in a sibling
working dir under --out-dir.

Usage:
    python -m surya.scripts.build_gguf \\
        --checkpoint datalab-to/surya-2.1.0 \\
        --out-dir ./gguf-build
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp.git"
# Pinned commit the converter patches were authored against. Bump when
# the upstream converter drifts; re-validate the anchor strings below.
LLAMA_CPP_REV = "bbeb89d76c41bc250f16e4a6fefcc9b530d6e3f3"


# ---- llama.cpp converter patches -----------------------------------------
# Each patch is (sentinel, anchor, replacement). Idempotent: if `sentinel`
# is already in the file, the patch is skipped.

_PATCH_REGISTER_HASH_ANCHOR = """        if chkhsh == "862f827721df956049dff5ca81a57f29e575280bc622e290d3bf4e35eca29015":
            # ref: https://huggingface.co/codefuse-ai/F2LLM-v2-4B
            res = "f2llmv2"
"""

_PATCH_REGISTER_HASH_REPLACEMENT = (
    _PATCH_REGISTER_HASH_ANCHOR
    + """        if chkhsh == "11865354be60ff9206694aed04242190f1807029bbd750bfbced09b4d26f1ad2":
            # surya-2.1.0 char-level WordLevel tokenizer (Split regex=".")
            res = "default"
"""
)

_PATCH_VOCAB_ANCHOR = """    def _set_vocab_gpt2(self) -> None:
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)
"""

_PATCH_VOCAB_REPLACEMENT = '''    def _set_vocab_gpt2(self) -> None:
        tokens, toktypes, tokpre = self.get_vocab_base()
        # surya: char-level / WordLevel vocabs contain raw bytes (e.g. " ",
        # "\\n"). llama.cpp's gpt2 vocab decoder applies bytes_to_unicode
        # when emitting NORMAL tokens, so encode bytes here for round-trip.
        # Idempotent on already-encoded vocabs.
        tokens = self._maybe_encode_gpt2_bytes(tokens, toktypes)
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)
        # surya: char-level / WordLevel vocabs have no merges, but the gpt2
        # vocab loader requires the field. Write a single dummy entry.
        if not special_vocab.merges:
            self.gguf_writer.add_token_merges(["a a"])

    @staticmethod
    def _maybe_encode_gpt2_bytes(tokens: list[str], toktypes: list[int]) -> list[str]:
        """Apply GPT-2 bytes_to_unicode to NORMAL tokens iff any contain raw
        whitespace/control bytes. Idempotent on already-encoded vocabs."""
        bs = (list(range(ord("!"), ord("~") + 1))
              + list(range(ord("¡"), ord("¬") + 1))
              + list(range(ord("®"), ord("ÿ") + 1)))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(2 ** 8 + n)
                n += 1
        byte_to_unicode = {b: chr(c) for b, c in zip(bs, cs)}
        printable = set(range(ord("!"), ord("~") + 1))
        needs = False
        for tok, ttype in zip(tokens, toktypes):
            if ttype != gguf.TokenType.NORMAL:
                continue
            for ch in tok:
                cb = ord(ch)
                if cb < 0x80 and cb not in printable:
                    needs = True
                    break
            if needs:
                break
        if not needs:
            return tokens
        out: list[str] = []
        for tok, ttype in zip(tokens, toktypes):
            if ttype == gguf.TokenType.NORMAL:
                out.append("".join(byte_to_unicode[b] for b in tok.encode("utf-8")))
            else:
                out.append(tok)
        return out
'''

CONVERTER_PATCHES = [
    {
        "name": "register surya-2.1.0 pre-tokenizer hash",
        "sentinel": '"11865354be60ff9206694aed04242190f1807029bbd750bfbced09b4d26f1ad2"',
        "anchor": _PATCH_REGISTER_HASH_ANCHOR,
        "replacement": _PATCH_REGISTER_HASH_REPLACEMENT,
    },
    {
        "name": "byte-encode NORMAL tokens + dummy merges in _set_vocab_gpt2",
        "sentinel": "_maybe_encode_gpt2_bytes",
        "anchor": _PATCH_VOCAB_ANCHOR,
        "replacement": _PATCH_VOCAB_REPLACEMENT,
    },
]


def patch_converter(convert_py: Path) -> None:
    text = convert_py.read_text()
    changed = False
    for p in CONVERTER_PATCHES:
        if p["sentinel"] in text:
            print(f"  [skip]  {p['name']} (already applied)")
            continue
        if p["anchor"] not in text:
            raise RuntimeError(
                f"converter patch {p['name']!r} could not find its anchor. "
                f"llama.cpp upstream may have drifted; re-pin LLAMA_CPP_REV "
                f"and re-validate anchors."
            )
        text = text.replace(p["anchor"], p["replacement"], 1)
        print(f"  [apply] {p['name']}")
        changed = True
    if changed:
        convert_py.write_text(text)


# ---- checkpoint-side patches ----------------------------------------------


def patch_checkpoint(src: Path, dst: Path) -> None:
    """Symlink-clone src into dst, with config.json + tokenizer_config.json
    rewritten for stock transformers/llama.cpp compatibility."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    overrides = {"config.json", "tokenizer_config.json"}
    for entry in src.iterdir():
        if entry.name in overrides:
            continue
        os.symlink(entry.resolve(), dst / entry.name)

    cfg = json.loads((src / "config.json").read_text())
    cfg["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    tk = json.loads((src / "tokenizer_config.json").read_text())
    tk["tokenizer_class"] = "PreTrainedTokenizerFast"
    for k in ("backend", "extra_special_tokens", "is_local"):
        tk.pop(k, None)
    (dst / "tokenizer_config.json").write_text(json.dumps(tk, indent=2))


# ---- llama.cpp resolution -------------------------------------------------


def ensure_llama_cpp(repo_dir: Path, rev: str) -> Path:
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[clone] {LLAMA_CPP_REPO} → {repo_dir}")
        subprocess.check_call(["git", "clone", LLAMA_CPP_REPO, str(repo_dir)])
    head = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != rev:
        # Discard any prior patches so the checkout is clean.
        subprocess.check_call(
            ["git", "-C", str(repo_dir), "reset", "--hard", "--quiet", "HEAD"]
        )
        subprocess.check_call(
            ["git", "-C", str(repo_dir), "fetch", "--quiet", "origin"]
        )
        print(f"[checkout] llama.cpp @ {rev}")
        subprocess.check_call(["git", "-C", str(repo_dir), "checkout", "--quiet", rev])
    return repo_dir


# ---- checkpoint resolution ------------------------------------------------


def resolve_checkpoint(checkpoint: str) -> Path:
    p = Path(checkpoint)
    if p.exists():
        return p.resolve()
    from huggingface_hub import snapshot_download

    print(f"[download] {checkpoint}")
    return Path(snapshot_download(checkpoint))


# ---- main -----------------------------------------------------------------


def main() -> int:
    from surya.settings import settings

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--checkpoint",
        default=settings.SURYA_MODEL_CHECKPOINT,
        help="HF repo id or local checkpoint dir",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("./gguf-build"))
    ap.add_argument(
        "--name",
        default="surya-2",
        help="Output basename. Produces <name>.gguf and <name>-mmproj.gguf",
    )
    ap.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=Path.home() / ".cache" / "datalab" / "llama.cpp",
    )
    ap.add_argument("--llama-cpp-rev", default=LLAMA_CPP_REV)
    ap.add_argument(
        "--outtype",
        default="f16",
        help="convert_hf_to_gguf --outtype (f16, bf16, q8_0, ...)",
    )
    ap.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep the patched-checkpoint working dir on success",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    work = args.out_dir / "_patched_ckpt"

    src = resolve_checkpoint(args.checkpoint)
    print(f"[checkpoint] {src}")

    print("[patch] checkpoint config + tokenizer_config")
    patch_checkpoint(src, work)

    repo = ensure_llama_cpp(args.llama_cpp_dir, args.llama_cpp_rev)
    convert_py = repo / "convert_hf_to_gguf.py"
    print("[patch] llama.cpp convert_hf_to_gguf.py")
    patch_converter(convert_py)

    out_llm = (args.out_dir / f"{args.name}.gguf").resolve()
    out_mmproj = (args.out_dir / f"{args.name}-mmproj.gguf").resolve()

    print(f"[convert] LLM → {out_llm}")
    subprocess.check_call(
        [
            sys.executable,
            str(convert_py),
            str(work),
            "--outfile",
            str(out_llm),
            "--outtype",
            args.outtype,
        ]
    )
    print(f"[convert] mmproj → {out_mmproj}")
    subprocess.check_call(
        [
            sys.executable,
            str(convert_py),
            str(work),
            "--mmproj",
            "--outfile",
            str(out_mmproj),
            "--outtype",
            args.outtype,
        ]
    )

    if not args.keep_work:
        shutil.rmtree(work)

    print()
    print(f"  LLM:    {out_llm}")
    print(f"  mmproj: {out_mmproj}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
