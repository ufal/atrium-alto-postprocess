"""
alto_stats_create.py

Purpose:
This script scans a given input folder for ALTO XML files. It can scan
both the root of the folder and one level of subdirectories.

For each ALTO XML file found, it executes the external command 'alto-tools -s'
(statistics) to get counts of various XML elements (e.g., <TextLine>,
<String>, <Illustration>).

It then parses this output and compiles all the statistics into a single
CSV file, along with the file/page identifiers derived from the filenames
and the full path to the XML file.

This CSV is the primary input for the next step in the pipeline
(e.g., run_langID.py).

Dependencies:
- alto-tools (must be installed and in the system's PATH)
- pandas (Python library)

Usage:
    python alto_stats_create.py <input_folder> [-o <output_csv>]

Example:
    python alto_stats_create.py ./my_alto_files/ -o stats.csv
"""

import os
import argparse
import subprocess  # To run external commands (like alto-tools)
import pandas as pd  # To easily create the final CSV
import re  # For regular expressions, to parse the command output
from atrium_paradata import ParadataLogger
import sys

def parse_alto_tools_stats_line(line):
    """
    Parses a single line of output from `alto-tools -s`.

    Example input line:
      "# of <TextLine> elements: 33"

    Example output dict:
      {"textlines": 33}

    Args:
        line (str): A single line of text from the command output.

    Returns:
        dict or None: A dictionary with a normalized key (e.g., "textlines")
                      and the integer count, or None if the line doesn't match.
    """
    # This regex looks for:
    #   "# of <" + (one or more word characters) + "> elements:" + (optional whitespace) + (one or more digits)
    m = re.match(r"# of <(\w+)> elements:\s+(\d+)", line.strip())

    if not m:
        # Line didn't match the pattern (e.g., it's an empty line)
        return None

    # m.groups() will be ("TextLine", "33")
    element, count = m.groups()
    element = element.lower()  # Normalize to lowercase (e.g., "textline")

    # Map from the XML element name to the desired CSV column name
    mapping = {
        "textline": "textlines",
        "string": "strings",
        "glyph": "glyphs",
        "illustration": "illustrations",
        "graphicalelement": "graphics",
    }

    # Use the mapped name if it exists, otherwise just use the element name
    key = mapping.get(element, element)
    return {key: int(count)}


def run_alto_tools_stats(xml_path):
    """
    Runs the `alto-tools -s` command on a single XML file and parses its output.

    Args:
        xml_path (str): The full path to the ALTO XML file.

    Returns:
        dict or None: A dictionary containing all statistics for the file,
                      or None if the command fails.
    """
    cmd = ["alto-tools", "-s", xml_path]
    try:
        # Run the command and capture its standard output
        # 'stderr=subprocess.STDOUT' merges error messages into the output
        # 'text=True' decodes the output as text (not bytes)
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        # The command failed (returned a non-zero exit code)
        print(f"⚠️ Error running alto-tools on {xml_path}: {e.output}")
        return None

    stats = {}
    # Process the command's output line by line
    for line in out.splitlines():
        parsed = parse_alto_tools_stats_line(line)
        if parsed:
            # Add the parsed {key: value} to our main stats dictionary
            stats.update(parsed)
    return stats


def process_alto_files_with_alto_tools(directory_path):
    """
    Processes all ALTO XML files found directly within a given directory.

    Args:
        directory_path (str): The folder to scan for .xml files.

    Returns:
        list[dict]: A list of dictionaries, where each dict holds the
                    stats for one file.
    """
    results = []
    # Loop through every file in the directory
    _total_inputs = 0
    _skips = []
    for fname in os.listdir(directory_path):
        # Skip files that don't end in .xml
        if not fname.lower().endswith(".xml"):
            continue

        xml_path = os.path.join(directory_path, fname)
        _total_inputs += 1


        # Get the statistics for this file
        stats = run_alto_tools_stats(xml_path)
        if stats is None:
            _skips.append(xml_path)
            # An error occurred and was already printed, so just skip this file
            continue

        # --- Derive file ID and page ID from the filename ---
        # e.g., "doc123-001.alto.xml"
        base = os.path.basename(fname).split(".")[0]  # "doc123-001"
        parts = base.split("-")  # ["doc123", "001"]
        file_id = parts[0]  # "doc123"
        page = parts[1] if len(parts) > 1 else ""  # "001"

        # Build the result dictionary for this file
        rec = {
            "file": file_id,
            "page": page,
        }

        # Map the parsed keys to our final dictionary keys, defaulting to 0
        rec["textlines"] = int(stats.get("textlines", 0))
        rec["illustrations"] = int(stats.get("illustrations", 0))
        rec["graphics"] = int(stats.get("graphics", 0))
        rec["strings"] = int(stats.get("strings", 0))
        # Add the full path, as this is needed by later scripts
        rec["path"] = xml_path

        results.append(rec)
    return results, _total_inputs, _skips


def main():
    # --- 1. Setup Argument Parser ---
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="Folder containing ALTO XML files or subfolders with them")
    parser.add_argument("-o", "--output", default="alto_stats.csv", help="Output CSV file path")
    args = parser.parse_args()

    # --- 2. Prepare Output File ---
    # Remove the output file if it already exists, so we start fresh
    if os.path.exists(args.output):
        os.remove(args.output)

    # --- 3. Find Subdirectories ---
    # This script is designed to check the root input_folder *and*
    # one level of subdirectories.
    subdirs = [os.path.join(args.input_folder, d)
               for d in os.listdir(args.input_folder)
               if os.path.isdir(os.path.join(args.input_folder, d))]

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
    )
    _total_inputs = 0

    try:
        # --- 4. Process Subdirectories ---
        for subdir in subdirs:
            stats, doc_inputs, doc_skips = process_alto_files_with_alto_tools(subdir)
            _total_inputs += doc_inputs
            _logger.log_success("csv", count=len(stats))
            for sk in doc_skips:
                _logger.log_skip(sk, "alto-tools failed to parse this file")
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
            _logger.log_skip(sk, "alto-tools failed to parse this file")

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