#!/usr/bin/env python3
"""
page_split.py

Purpose:
This script takes a multi-page ALTO XML file as input and splits it into
multiple single-page ALTO XML files. Each output file will contain the
full header and style information from the original file, but the <Layout>
section will only contain the data for a single page.

Usage:
    python page_split.py <input_directory> <output_directory>
"""

import argparse
import os
import xml.etree.ElementTree as ET  # For parsing and creating XML

from atrium_paradata import ParadataLogger


def _make_safe_parser() -> ET.XMLParser:
    """Build an ElementTree parser that does not resolve external entities.

    (#5) ALTO documents may be untrusted. stdlib ElementTree does not expand
    external entities by default, but we attach an explicit handler that refuses
    any entity declaration so a crafted DOCTYPE cannot trigger entity-expansion
    ("billion laughs") blow-ups.
    """
    parser = ET.XMLParser()
    try:
        # expat: disable entity definitions and external entity resolution.
        parser.parser.DefaultHandler = lambda data: None
        parser.parser.EntityDeclHandler = lambda *a, **k: (_ for _ in ()).throw(
            ET.ParseError("entities are not allowed")
        )
        parser.parser.ExternalEntityRefHandler = lambda *a, **k: False
    except AttributeError:
        # Some Python builds expose a restricted expat; fall back to the default
        # parser, which still does not resolve external entities.
        pass
    return parser


def split_alto_xml(input_file_path, output_dir):
    """
    Splits a single multi-page ALTO XML file into single-page files.

    Returns:
        int: The number of pages written (0 if no pages were found).
    """
    namespace = {"alto": "http://www.loc.gov/standards/alto/ns-v3#"}
    ET.register_namespace("", "http://www.loc.gov/standards/alto/ns-v3#")

    # --- Parse the Input XML (hardened parser) ---
    tree = ET.parse(input_file_path, parser=_make_safe_parser())
    root = tree.getroot()

    description = root.find("alto:Description", namespace)
    styles = root.find("alto:Styles", namespace)

    pages = root.findall(".//alto:Page", namespace)

    if not pages:
        print(f"  -> No <Page> elements found in {input_file_path}. Skipping.")
        return 0

    base_name = os.path.splitext(os.path.basename(input_file_path))[0].replace(".alto", "")

    page_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(page_output_dir, exist_ok=True)

    print(f"  -> Found {len(pages)} page(s). Splitting...")
    for i, page in enumerate(pages, 1):
        page_number = page.get("PHYSICAL_IMG_NR", str(i))
        output_filename = f"{base_name}-{page_number}.alto.xml"
        output_filepath = os.path.join(page_output_dir, output_filename)

        new_root = ET.Element(root.tag, root.attrib)
        if description is not None:
            new_root.append(description)
        if styles is not None:
            new_root.append(styles)

        new_layout = ET.SubElement(new_root, "Layout")
        new_layout.append(page)

        new_tree = ET.ElementTree(new_root)
        new_tree.write(output_filepath, encoding="UTF-8", xml_declaration=True)

    print(f"  -> Successfully split into {len(pages)} file(s) in '{page_output_dir}'.")
    return len(pages)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Split multi-page ALTO XML files into single-page files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input_dir", help="Path to the directory containing ALTO XML files to process.")
    parser.add_argument("output_dir", help="Path to the directory where split files will be saved.")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at '{args.input_dir}'")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to '{os.path.abspath(args.output_dir)}'\n")

    _logger = ParadataLogger(
        program="alto-postprocess",
        config={
            "script": "page_split",
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
        },
        paradata_dir="paradata",
        output_types=["xml"],
    )

    # (#10) Track documents (the unit of input) and pages (the unit of output)
    # separately. input_files_total counts source documents; the per-page count
    # feeds the "xml" output total for throughput; successfully_processed is the
    # number of documents successfully split — so it can never exceed inputs.
    _total_inputs = 0
    _docs_ok = 0

    try:
        for filename in sorted(os.listdir(args.input_dir)):
            if filename.lower().endswith(".xml"):
                input_file_path = os.path.join(args.input_dir, filename)
                print(f"Processing '{filename}'...")
                _total_inputs += 1
                try:
                    page_count = split_alto_xml(input_file_path, args.output_dir)
                    _logger.log_success("xml", count=page_count)  # pages produced
                    if page_count > 0:
                        _docs_ok += 1
                except Exception as e:
                    _logger.log_skip(str(filename), str(e))
    finally:
        _logger.finalize(input_total=_total_inputs, processed_total=_docs_ok)


if __name__ == "__main__":
    main()
