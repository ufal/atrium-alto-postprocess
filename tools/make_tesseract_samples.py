#!/usr/bin/env python3
"""
tools/make_tesseract_samples.py — regenerate the real Tesseract samples of data_samples/TEXT/ (#31 Phase 5).

Renders two pages of an invented Czech excavation report (the fictional world of
data_samples/: the site "Hradiště u Horní Mezí", Jan Novotný, Eva Procházková) with
Pillow and DejaVu Serif, OCRs them with the `tesseract` CLI and writes its output:

  CTX000000025.tsv   Tesseract TSV (level 1-5 rows), 2 pages
  CTX000000026.hocr  Tesseract hOCR, 2 pages

The OCR output is the engine's own, byte for byte, except that the absolute path of the
rendered page images (in `title="image ..."` and the TSV has none) is replaced by the
bare file name. It depends on the Tesseract version and its `ces` model, so the files are
committed rather than rebuilt in tests; tests/test_text_formats.py pins what the readers
make of them. Needs `tesseract` with the `ces` language (apt: tesseract-ocr
tesseract-ocr-ces), Pillow, and the DejaVu fonts.

Usage:
    python3 tools/make_tesseract_samples.py [--out data_samples/TEXT]
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
SIZE = (1700, 2200)  # A4 at 200 dpi
PAGES = [
    [
        ("DejaVuSerif-Bold.ttf", 44, "HRADIŠTĚ U HORNÍ MEZÍ"),
        ("DejaVuSerif-Bold.ttf", 34, "Zpráva o archeologickém výzkumu 2024"),
        None,
        ("DejaVuSerif.ttf", 30, "Sonda II byla vytyčena v červnu 2024 na severním"),
        ("DejaVuSerif.ttf", 30, "okraji plošiny, kde se v roce 2019 objevily zlomky"),
        ("DejaVuSerif.ttf", 30, "keramiky a mazanice. Sonda o rozměrech 4 × 2 m"),
        ("DejaVuSerif.ttf", 30, "zachytila vnitřní líc valu a dvě sídlištní vrstvy."),
        None,
        ("DejaVuSerif.ttf", 30, "Nálezy: střepy nádob, přeslen, zvířecí kosti a želez-"),
        ("DejaVuSerif.ttf", 30, "ný nůž. Všechny nálezy byly uloženy v depozitáři"),
        ("DejaVuSerif.ttf", 30, "muzea v Horní Mezi pod přírůstkovým číslem 17/2024."),
        None,
        ("DejaVuSerif.ttf", 26, "— 1 —"),
    ],
    [
        ("DejaVuSerif-Bold.ttf", 34, "Stratigrafie sondy II"),
        None,
        ("DejaVuSerif.ttf", 30, "Vrstva 1: ornice, tmavě hnědá hlína s kořeny."),
        ("DejaVuSerif.ttf", 30, "Vrstva 2: šedohnědá hlína s uhlíky a mazanicí."),
        ("DejaVuSerif.ttf", 30, "Vrstva 3: žlutý jíl, podloží, bez nálezů."),
        None,
        ("DejaVuSerif.ttf", 30, "Výzkum vedl Jan Novotný, kresby zhotovila"),
        ("DejaVuSerif.ttf", 30, "Eva Procházková. Děkujeme obci Horní Mez."),
        None,
        ("DejaVuSerif.ttf", 26, "— 2 —"),
    ],
]
OUTPUTS = {"tsv": "CTX000000025.tsv", "hocr": "CTX000000026.hocr"}


def render(workdir: str) -> list:
    from PIL import Image, ImageDraw, ImageFont

    paths = []
    for n, lines in enumerate(PAGES, 1):
        img = Image.new("L", SIZE, 255)
        draw = ImageDraw.Draw(img)
        y = 180
        for item in lines:
            if item is None:
                y += 40
                continue
            font_name, size, text = item
            font = ImageFont.truetype(os.path.join(FONT_DIR, font_name), size)
            x = (SIZE[0] - draw.textlength(text, font=font)) / 2 if text.startswith("—") else 160
            draw.text((x, y), text, font=font, fill=0)
            y += int(size * 1.6)
        path = os.path.join(workdir, f"page{n}.png")
        img.save(path, dpi=(200, 200))
        paths.append(path)
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--out", default=os.path.join("data_samples", "TEXT"), help="Output directory.")
    args = parser.parse_args(argv)
    if shutil.which("tesseract") is None:
        print("tesseract is not installed (apt: tesseract-ocr tesseract-ocr-ces)", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as workdir:
        images = render(workdir)
        listing = os.path.join(workdir, "pages.txt")
        with open(listing, "w", encoding="utf-8") as fh:
            fh.write("\n".join(images) + "\n")
        base = os.path.join(workdir, "report")
        subprocess.run(
            ["tesseract", listing, base, "-l", "ces", "--dpi", "200", *OUTPUTS], check=True, capture_output=True
        )
        os.makedirs(args.out, exist_ok=True)
        for ext, name in OUTPUTS.items():
            with open(f"{base}.{ext}", encoding="utf-8") as fh:
                text = fh.read()
            for image in images:
                text = text.replace(image, os.path.basename(image))
            with open(os.path.join(args.out, name), "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
            print(f"wrote {os.path.join(args.out, name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
