#!/usr/bin/env python3
"""Release preflight: everything that must hold before publishing.

    uv run python scripts/preflight.py --hf ../hf_mon_tokenizer

Publishing is irreversible — a PyPI version can be yanked but never reused, and a
Hugging Face repo is what people load by default. The previous release shipped
three defects that no test would have caught because no test looked at the
artifact: `twine` as a runtime dependency, a version string four releases stale,
and a `tokenizer.json` on Hugging Face that was the 4,000-piece predecessor while
the card advertised 32,000.

Every check here inspects the **built artifact or a clean install**, not the
source tree. A check that passes because the repository happens to be on
`sys.path` proves nothing about what a user receives.

Exit code is the number of failures, so this composes in a shell chain.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent

PROBES = [
    "ဂွံအခေါင်အရာမွဲသ္ဂောံဒုင်စသိုင်ကၠာကၠာရ။",
    "မြန်မာနိုင်ငံသည် အရှေ့တောင်အာရှတွင် တည်ရှိသော နိုင်ငံဖြစ်သည်။",
    "The Mon language is spoken by about one million people.",
    "ကျော် page 42 — “quoted” ၏ 🙏 ภาษา ə café ×2 …",
    "🙏 emoji",
    "ภาษาไทย",
    "漢字とかな",
    "a   b",  # whitespace runs are content
]

results: list[tuple[bool, str]] = []


def check(name: str, ok: bool, detail: str = "", on_pass: str = "") -> None:
    """Record a result. `detail` is shown only on FAILURE.

    An earlier version appended it unconditionally, so a passing check printed
    `PASS hf ids identical to package — differs on at least one probe`. A report
    that contradicts itself is worse than no report: it teaches the reader to
    skim past the words and trust only the PASS/FAIL, which is exactly how a real
    failure gets missed. Pass `on_pass` for detail worth showing when green.
    """
    suffix = detail if not ok else on_pass
    results.append((ok, f"{name}{' — ' + suffix if suffix else ''}"))


def run(command: list[str], cwd: Path = ROOT) -> tuple[int, str]:
    p = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", type=Path, help="Hugging Face repo checkout to verify too")
    args = parser.parse_args()

    # -- the local gate ----------------------------------------------------
    for name, command in [
        ("ruff format", ["uv", "run", "ruff", "format", "--check", "."]),
        ("ruff check", ["uv", "run", "ruff", "check", "."]),
        ("mypy", ["uv", "run", "mypy"]),
        ("pytest", ["uv", "run", "pytest", "-q"]),
    ]:
        code, out = run(command)
        check(name, code == 0, "" if code == 0 else out.strip().splitlines()[-1][:90])

    # -- the artifact, built and inspected ---------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        code, out = run(["uv", "build", "--wheel", "--out-dir", tmp])
        check("wheel builds", code == 0, "" if code == 0 else out.strip()[-90:])
        wheels = list(Path(tmp).glob("*.whl"))
        check("wheel produced", bool(wheels))

        if wheels:
            with zipfile.ZipFile(wheels[0]) as archive:
                names = archive.namelist()
                art = [n for n in names if n.endswith("mon_tokenizer.json")]
                card = [n for n in names if n.endswith("model_card.json")]
                check("artifact in wheel", bool(art))
                check("model card in wheel", bool(card))
                if art:
                    size = archive.getinfo(art[0]).file_size
                    check("artifact not truncated", size > 4_000_000, on_pass=f"{size:,} bytes")
                # A SentencePiece model must not ship: 1.0.0 has one artifact, and
                # two would be two things that can disagree.
                check("no stale .model", not [n for n in names if n.endswith(".model")])

            # A CLEAN environment. `--no-project` keeps the source tree off
            # sys.path, so a missing data file cannot be masked by it.
            probe = json.dumps(PROBES, ensure_ascii=False)
            script = (
                "import json,sys\n"
                "from mon_tokenizer import MonTokenizer, model_card, __version__\n"
                "t=MonTokenizer(); c=model_card()\n"
                f"probes=json.loads(r'''{probe}''')\n"
                "out={'vocab':t.get_vocab_size(),'version':__version__,"
                "'card_vocab':c['vocab_size'],"
                "'artifact_version':c.get('artifact_version'),"
                "'card_digest':c['corpus']['source_digest'],"
                "'card_has_bare_version':'version' in c,"
                "'specials':[t.unk_id,t.bos_id,t.eos_id,t.pad_id],"
                "'bytes':sum(1 for v in range(256) if f'<0x{v:02X}>' in t.get_vocab()),"
                "'rt':[t.decode_ids(t.encode(p)['ids'])==t.encode(p)['text'] for p in probes],"
                "'ids':[t.encode_ids(p) for p in probes]}\n"
                "print('PREFLIGHT'+json.dumps(out))\n"
            )
            code, out = run(
                [
                    "uv",
                    "run",
                    "--isolated",
                    "--no-project",
                    "--with",
                    str(wheels[0]),
                    "python",
                    "-c",
                    script,
                ]
            )
            marker = [ln for ln in out.splitlines() if ln.startswith("PREFLIGHT")]
            check("clean install imports", bool(marker), "" if marker else out.strip()[-90:])
            if marker:
                data = json.loads(marker[0][len("PREFLIGHT") :])
                check("vocab 64,256", data["vocab"] == 64_256, str(data["vocab"]))
                check("card agrees with artifact", data["card_vocab"] == data["vocab"])
                # The card was checked for vocab only, which is how a hardcoded
                # `"version": "1.0.0"` sat in it ungated. It names the trained
                # model, so it must be present and must never silently revert to
                # a bare `version` that reads as the package's.
                check(
                    "card names its artifact version",
                    bool(data["artifact_version"]),
                    on_pass=str(data["artifact_version"]),
                )
                check("card has no bare 'version' key", not data["card_has_bare_version"])
                check(
                    "specials are 0,1,2,3", data["specials"] == [0, 1, 2, 3], str(data["specials"])
                )
                check("all 256 byte pieces", data["bytes"] == 256, on_pass=f"{data['bytes']}/256")
                check(
                    "every probe round-trips",
                    all(data["rt"]),
                    on_pass=f"{sum(data['rt'])}/{len(data['rt'])}",
                )
                check(
                    "version matches manifest",
                    data["version"] == _manifest_version(),
                    on_pass=data["version"],
                )
                # -- Hugging Face parity ----------------------------------
                if args.hf:
                    _check_hf(args.hf, data["ids"], data["artifact_version"], data["card_digest"])

    failures = [name for ok, name in results if not ok]
    width = max(len(n) for _, n in results)
    print()
    for ok, name in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}")
    print(f"\n{len(results) - len(failures)}/{len(results)} checks passed")
    if failures:
        print("\nDO NOT PUBLISH. Failed:")
        for name in failures:
            print(f"  - {name}")
    else:
        print("\nSafe to publish.")
    return len(failures)


def _manifest_version() -> str:
    import tomllib

    return str(
        tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    )


def _check_hf(
    hf: Path, package_ids: list[list[int]], artifact_version: str | None, card_digest: str
) -> None:
    """The Hugging Face repo must produce the same ids as the package.

    This is the check that would have caught the shipped defect: `tokenizer.json`
    on the Hub was the 4,000-piece predecessor while `tokenizer.model` was the
    32,000 one, and `AutoTokenizer` prefers the former. Anyone using the
    documented path got a tokenizer measuring 0.93 chars/token against an
    advertised 5.22.
    """
    check("hf repo exists", hf.exists(), str(hf))
    if not hf.exists():
        return
    for required in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        check(f"hf has {required}", (hf / required).exists())
    check("hf has no .model", not list(hf.glob("*.model")))

    # Two copies of one document. They are published to different places by
    # different commands, so nothing but this makes them agree.
    hf_card = hf / "model_card.json"
    check("hf has model_card.json", hf_card.exists())
    if hf_card.exists():
        card = json.loads(hf_card.read_text(encoding="utf-8"))
        check(
            "hf card artifact version matches package",
            card.get("artifact_version") == artifact_version,
            f"hf {card.get('artifact_version')!r} vs package {artifact_version!r}",
        )
        check(
            "hf card corpus digest matches package",
            card.get("corpus", {}).get("source_digest") == card_digest,
            "the two cards describe different corpora",
        )

    # Exit code is not checked: a non-zero exit shows up as a missing marker,
    # which the next check reports with the actual output attached.
    _, out = run(
        [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--with",
            "transformers",
            "--with",
            "tokenizers",
            "python",
            "-c",
            "import json,sys\n"
            "from transformers import AutoTokenizer\n"
            f"t=AutoTokenizer.from_pretrained(r'{hf}')\n"
            f"probes=json.loads(r'''{json.dumps(PROBES, ensure_ascii=False)}''')\n"
            "print('HF'+json.dumps({'vocab':len(t),"
            "'ids':[t.encode(p, add_special_tokens=False) for p in probes]}))\n",
        ]
    )
    marker = [ln for ln in out.splitlines() if ln.startswith("HF")]
    check("hf loads via AutoTokenizer", bool(marker), "" if marker else out.strip()[-90:])
    if marker:
        data = json.loads(marker[0][2:])
        check("hf vocab 64,256", data["vocab"] == 64_256, str(data["vocab"]))
        check(
            "hf ids identical to package",
            data["ids"] == package_ids,
            "differs on at least one probe",
        )


if __name__ == "__main__":
    raise SystemExit(main())
