#!/usr/bin/env python3
"""
page_split.py

Purpose:
This script takes a multi-page ALTO XML file as input and splits it into
multiple single-page ALTO XML files. Each output file will contain the
full header and style information from the original file, but the <Layout>
section will only contain the data for a single page.

This is useful for systems that process ALTO files on a per-page basis.

Usage:
    python page_split.py <input_directory> <output_directory>

Example:
    python page_split.py ./my_multi_page_altos/ ./my_single_page_altos/
"""

# Import necessary libraries
import xml.etree.ElementTree as ET  # For parsing and creating XML
import os  # For file and directory operations (paths, mkdir)
import argparse  # For parsing command-line arguments
from atrium_paradata import ParadataLogger


def split_alto_xml(input_file_path, output_dir):
    """
    Splits a single multi-page ALTO XML file into single-page files.

    Each new file contains the full header (Description, Styles) and the
    Layout for a single page, saved into the specified output directory.

    Args:
        input_file_path (str): The full path to the ALTO XML file to be split.
        output_dir (str): The directory where split files will be saved.
                          A subdirectory will be created inside here named
                          after the base name of the input file.
    """
    try:
        # --- 1. Setup XML Namespace ---
        # ALTO XML uses a specific namespace. We must register it to
        # make sure ElementTree can find the elements (e.g., <Page>, <Styles>).
        # We also register it as the *default* namespace ('') so that
        # when we write the new XML files, it doesn't add "ns0:" prefixes.
        namespace = {'alto': 'http://www.loc.gov/standards/alto/ns-v3#'}
        ET.register_namespace('', 'http://www.loc.gov/standards/alto/ns-v3#')

        # --- 2. Parse the Input XML ---
        tree = ET.parse(input_file_path)
        root = tree.getroot()

        # --- 3. Find Common Header Elements ---
        # We need to copy the <Description> and <Styles> blocks into
        # *every* new single-page file so they are all valid.
        description = root.find('alto:Description', namespace)
        styles = root.find('alto:Styles', namespace)

        # --- 4. Find All Page Elements ---
        # Find all <Page> elements anywhere within the <Layout> section.
        pages = root.findall('.//alto:Page', namespace)

        if not pages:
            print(f"  -> No <Page> elements found in {input_file_path}. Skipping.")
            return

        # --- 5. Prepare Output Directory and Filename ---
        # Get the base name of the input file (e.g., "document1.alto.xml" -> "document1")
        base_name = os.path.splitext(os.path.basename(input_file_path))[0].replace(".alto", "")

        # Create a dedicated subdirectory for this document's pages
        # (e.g., output_dir/document1/)
        page_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(page_output_dir, exist_ok=True)

        # --- 6. Loop Through Pages and Create New Files ---
        print(f"  -> Found {len(pages)} page(s). Splitting...")
        for i, page in enumerate(pages, 1):
            # Try to get the page number from the 'PHYSICAL_IMG_NR' attribute.
            # If it's not there, just use a simple counter (1, 2, 3...).
            page_number = page.get('PHYSICAL_IMG_NR', str(i))

            # Create the new filename (e.g., "document1-1.alto.xml")
            output_filename = f"{base_name}-{page_number}.alto.xml"
            output_filepath = os.path.join(page_output_dir, output_filename)

            # --- 7. Build the New XML Tree for This Page ---
            # Create a new root <alto> element
            new_root = ET.Element(root.tag, root.attrib)

            # Append the (optional) header elements we found earlier
            if description is not None:
                new_root.append(description)
            if styles is not None:
                new_root.append(styles)

            # Create a new <Layout> element...
            new_layout = ET.SubElement(new_root, 'Layout')
            # ...and append *only* the current page to it.
            new_layout.append(page)

            # --- 8. Write the New XML to a File ---
            new_tree = ET.ElementTree(new_root)
            new_tree.write(output_filepath, encoding='UTF-8', xml_declaration=True)

        print(f"  -> Successfully split into {len(pages)} file(s) in '{page_output_dir}'.")

    except ET.ParseError as e:
        print(f"  -> ERROR: Could not parse XML in {input_file_path}. {e}")
    except Exception as e:
        print(f"  -> ERROR: An unexpected error occurred: {e}")


def main():
    """
    Main function to handle command-line arguments and process files.
    """
    # --- 1. Setup Argument Parser ---
    # This defines the command-line arguments the script accepts.
    parser = argparse.ArgumentParser(
        description="Split multi-page ALTO XML files into single-page files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Define the 'input_dir' argument (required)
    parser.add_argument(
        "input_dir",
        help="Path to the directory containing ALTO XML files to process."
    )
    # Define the 'output_dir' argument (required)
    parser.add_argument(
        "output_dir",
        help="Path to the directory where split files will be saved. Will be created if it doesn't exist."
    )
    # Parse the arguments provided by the user
    args = parser.parse_args()

    # --- 2. Validate Input Directory ---
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at '{args.input_dir}'")
        return

    # --- 3. Create Output Directory ---
    # 'exist_ok=True' means it won't crash if the directory already exists.
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
    _total_inputs = 0

    # --- 4. Process Each XML File ---
    # Loop through all files in the input directory, sorted by name
    try:
        for filename in sorted(os.listdir(args.input_dir)):
            # Only process files that end with .xml (case-insensitive)
            if filename.lower().endswith('.xml'):
                input_file_path = os.path.join(args.input_dir, filename)
                print(f"Processing '{filename}'...")
                _total_inputs += 1

                # Call the main splitting function for this file
                try:
                    split_alto_xml(input_file_path, args.output_dir)
                    _logger.log_success("xml", count=page_count)
                except Exception as e:
                    _logger.log_skip(str(filename), str(e))
    finally:
        _logger.finalize(input_total=_total_inputs)

# --- Standard Python Entry Point ---
# This block checks if the script is being run directly (not imported).
# If it is, it calls the main() function.
if __name__ == "__main__":
    main()