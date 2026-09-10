"""
alto_stats_create.py

Purpose:
This script scans a given input folder for ALTO XML files. It can scan
both the root of the folder and one level of subdirectories.

For each ALTO XML file found, it counts various XML elements (e.g., <TextLine>,
<String>, <Illustration>) using the statistics code path vendored in
`alto_tools.py` — copied from https://github.com/cneud/alto-tools (Apache-2.0),
which is what the `alto-tools -s` CLI used to do in a subprocess.

It then compiles all the statistics into a single CSV file, along with the
file/page identifiers derived from the filenames and the full path to the XML
file.

This CSV is the primary input for the next step in the pipeline


Dependencies:
- alto_tools.py (vendored in this repository; no external binary, no PATH lookup)
- pandas (Python library)

Usage:
    python alto_stats_create.py <input_folder> [-o <output_csv>]

Example:
    python alto_stats_create.py ./my_alto_files/ -o stats.csv
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd  # To easily create the final CSV

import alto_tools  # Vendored `alto-tools -s` statistics path (issue #50)
from atrium_document import canonical_doc_id
from atrium_paradata import ParadataLogger


def run_alto_tools_stats(xml_path):
    """
    Counts the ALTO elements of a single XML file.

    (#50) This used to run `alto-tools -s <xml_path>` in a subprocess and regex
    its "# of <TextLine> elements: 33" stdout lines back into a dict. Both halves
    of that round-trip are gone: `alto_tools.statistics_from_file()` is the same
    upstream counting code, vendored into this repository, and it returns the
    dict directly under the keys this script writes to its CSV. The counts are
    identical to the CLI's — pinned by tests/test_alto_tools.py.

    Args:
        xml_path (str): The full path to the ALTO XML file.

    Returns:
        dict or None: A dictionary containing all statistics for the file
                      ("textlines", "strings", "glyphs", "illustrations",
                      "graphics"), or None if the file could not be read.
    """
    try:
        return alto_tools.statistics_from_file(xml_path)
    except Exception as e:
        # Same outcome as the old CalledProcessError branch: warn, skip the file.
        print(f"⚠️ Error reading ALTO statistics from {xml_path}: {e}")
        return None


def _process_single_xml(xml_path, fname):
    """
    Process one XML file: run alto-tools and build the result record.

    Returns:
        (dict, None)  on success — the record dict and no skip path.
        (None, str)   on failure — no record and the xml_path that was skipped.
    """
    stats = run_alto_tools_stats(xml_path)
    if stats is None:
        return None, xml_path

    # --- Derive file ID and page ID from the filename ---
    # e.g., "doc123-001.alto.xml"
    #
    # (atrium-project#10 D3) COMPOSED with the hub's canonical_doc_id() rather than
    # replaced by it: what page_split.py wrote is "<doc_id>-<page>.alto.xml", so this
    # site strips a pipeline suffix AND splits a page label off, and canonical_doc_id()
    # only does the first half (KNOWN_PIPELINE_SUFFIXES has no notion of a page
    # suffix). The old `split(".")[0]` also truncated any doc_id containing a dot —
    # "sbn.2019-1.alto.xml" became "sbn", so the stats CSV keyed a document that no
    # other stage had ever heard of. Suffix stripping now comes from the one shared
    # derivation; the page split stays local, because it is local knowledge.
    base = canonical_doc_id(os.path.basename(fname))  # "doc123-001"
    parts = base.split("-")  # ["doc123", "001"]
    file_id = parts[0]  # "doc123"
    page = parts[1] if len(parts) > 1 else ""  # "001"

    rec = {
        "file": file_id,
        "page": page,
        "textlines": int(stats.get("textlines", 0)),
        "illustrations": int(stats.get("illustrations", 0)),
        "graphics": int(stats.get("graphics", 0)),
        "strings": int(stats.get("strings", 0)),
        # Add the full path, as this is needed by later scripts
        "path": xml_path,
    }
    return rec, None


def process_alto_files_with_alto_tools(directory_path, max_workers=8):
    """
    Processes all ALTO XML files found directly within a given directory.

    Uses a ThreadPoolExecutor to overlap the per-file work. (#50) That work is
    now an in-process ElementTree parse rather than a spawned `alto-tools -s`
    subprocess, so the pool buys less than it did — but each file also no longer
    pays for an interpreter start-up, which dominated the old cost by far.

    Args:
        directory_path (str): The folder to scan for .xml files.
        max_workers (int):    Number of parallel threads (default: 8).

    Returns:
        tuple[list[dict], int, list[str]]:
            - list of per-file result dicts
            - total number of XML files found
            - list of xml_paths that failed (skipped)
    """
    xml_files = [
        (os.path.join(directory_path, fname), fname)
        for fname in os.listdir(directory_path)
        if fname.lower().endswith(".xml")
    ]

    _total_inputs = len(xml_files)
    results = []
    _skips = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(_process_single_xml, xml_path, fname): xml_path for xml_path, fname in xml_files
        }
        for future in as_completed(future_to_path):
            rec, skip_path = future.result()
            if skip_path is not None:
                _skips.append(skip_path)
            else:
                results.append(rec)

    return results, _total_inputs, _skips


def main(argv=None):
    # --- 1. Setup Argument Parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="Folder containing ALTO XML files or subfolders with them")
    parser.add_argument("-o", "--output", default="alto_stats.csv", help="Output CSV file path")
    args = parser.parse_args(argv)

    # --- 2. Prepare Output File ---
    # Remove the output file if it already exists, so we start fresh
    if os.path.exists(args.output):
        os.remove(args.output)

    # --- 3. Find Subdirectories ---
    # This script is designed to check the root input_folder *and*
    # one level of subdirectories.
    subdirs = [
        os.path.join(args.input_folder, d)
        for d in os.listdir(args.input_folder)
        if os.path.isdir(os.path.join(args.input_folder, d))
    ]

    # 'first' flag is used to ensure we only write the CSV header *once*
    first = True

    _logger = ParadataLogger(
        program="alto-postprocess",
        config={
            "script": "alto_stats_create",
            "input_dir": str(args.input_folder),
            "output_csv": str(args.output),
        },
        paradata_dir="paradata",
        output_types=["csv"],
        config_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup"),
    )
    _total_inputs = 0

    try:
        # --- 4. Process Subdirectories ---
        for subdir in subdirs:
            stats, doc_inputs, doc_skips = process_alto_files_with_alto_tools(subdir)
            _total_inputs += doc_inputs
            _logger.log_success("csv", count=len(stats))
            for sk in doc_skips:
                _logger.log_skip(sk, "alto-tools statistics failed to parse this file")
            if stats:
                # Convert the list of dictionaries into a pandas DataFrame
                df = pd.DataFrame(stats)
                if first:
                    # First write: include the header
                    df.to_csv(args.output, index=False, header=True)
                    first = False
                else:
                    # Subsequent writes: append (mode="a") and skip the header
                    df.to_csv(args.output, index=False, header=False, mode="a")
                print(f"Processed {len(stats)} files from {subdir}")

        # --- 5. Process Root Directory ---
        # After processing subdirs, process any .xml files in the root folder
        stats, doc_inputs, doc_skips = process_alto_files_with_alto_tools(args.input_folder)
        _total_inputs += doc_inputs
        _logger.log_success("csv", count=len(stats))
        for sk in doc_skips:
            _logger.log_skip(sk, "alto-tools statistics failed to parse this file")

        if stats:
            df = pd.DataFrame(stats)
            if first:
                df.to_csv(args.output, index=False, header=True)
                first = False
            else:
                df.to_csv(args.output, index=False, header=False, mode="a")
            print(f"Processed {len(stats)} files from {args.input_folder}")

        print("Done.")
    finally:
        _logger.finalize(input_total=_total_inputs)


if __name__ == "__main__":
    main()
