#!/usr/bin/env python3
"""Zero-dependency client for the ATRIUM ALTO Postprocessing API.

Uploads ALTO XML pages or plain-text files to a running instance of the
FastAPI service in `service/text_api.py` and returns per-line language and
quality classification (local server by default, remote via --base-url or the
ATRIUM_AP_URL env variable).

Only the Python 3 standard library is used - no pip installs required.

Usage:
    python3 scripts/atrium_postprocess.py page.alto.xml
    python3 scripts/atrium_postprocess.py page.txt --format csv
    python3 scripts/atrium_postprocess.py scans/*.xml --format json
    python3 scripts/atrium_postprocess.py notes.dat --task-type text
    python3 scripts/atrium_postprocess.py --info

Exit codes:
    0 - success
    1 - client-side error (bad arguments, unreadable file)
    2 - server unreachable (connection refused / timeout)
    3 - server-side error (HTTP 4xx/5xx)
"""

import argparse
import csv
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_BASE_URL = os.environ.get("ATRIUM_AP_URL", "http://localhost:8000")
MAX_UPLOAD_MB = 10  # mirrors the server's MAX_UPLOAD_MB default
RETRY_STATUS = {502, 503, 504}
RETRY_ATTEMPTS = 3
RETRY_WAIT_S = 10


def build_multipart(fields: dict, file_field: str, file_path: Path) -> tuple[bytes, str]:
    """Encode form fields and one file as multipart/form-data using only the stdlib."""
    boundary = uuid.uuid4().hex
    lines = []
    for name, value in fields.items():
        lines.append(f"--{boundary}".encode())
        lines.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        lines.append(b"")
        lines.append(str(value).encode())

    mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    lines.append(f"--{boundary}".encode())
    lines.append(f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"'.encode())
    lines.append(f"Content-Type: {mime}".encode())
    lines.append(b"")
    lines.append(file_path.read_bytes())
    lines.append(f"--{boundary}--".encode())
    lines.append(b"")

    body = b"\r\n".join(lines)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def http_json(url: str, data: bytes = None, content_type: str = None, timeout: int = 600) -> dict:
    """POST (or GET when data is None) and decode a JSON response, with retry on 502/503/504."""
    last_error = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
        if content_type:
            request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code in RETRY_STATUS and attempt < RETRY_ATTEMPTS:
                print(
                    f"[retry {attempt}/{RETRY_ATTEMPTS}] HTTP {e.code}, waiting {RETRY_WAIT_S}s...",
                    file=sys.stderr,
                )
                time.sleep(RETRY_WAIT_S)
                last_error = f"HTTP {e.code}: {detail}"
                continue
            print(f"Server error - HTTP {e.code}: {detail}", file=sys.stderr)
            sys.exit(3)
        except (urllib.error.URLError, TimeoutError) as e:
            print(
                f"Cannot reach the API at {url} ({e}).\nIs the server running? Start it with: bash scripts/server.sh",
                file=sys.stderr,
            )
            sys.exit(2)
    print(f"Server error after {RETRY_ATTEMPTS} attempts - {last_error}", file=sys.stderr)
    sys.exit(3)


def process_file(base_url: str, path: Path, task_type: str) -> dict:
    """Upload one file to POST /process."""
    size = path.stat().st_size
    if size > MAX_UPLOAD_MB * 1024 * 1024:
        print(
            f"Skipping {path}: {size} bytes exceeds the {MAX_UPLOAD_MB} MB server upload limit - "
            "split the document into single pages first",
            file=sys.stderr,
        )
        return {}
    if task_type == "auto" and path.suffix.lower() not in (".xml", ".txt"):
        print(
            f"Skipping {path}: cannot auto-detect type from suffix '{path.suffix}' - "
            "pass --task-type alto or --task-type text explicitly",
            file=sys.stderr,
        )
        return {}
    body, content_type = build_multipart({"task_type": task_type}, file_field="file", file_path=path)
    return http_json(f"{base_url}/process", data=body, content_type=content_type)


def result_rows(path: Path, result: dict) -> list[tuple]:
    """Flatten a /process response into (file, line, lang, quality, category, text) rows."""
    rows = []
    for line in result.get("lines", []):
        rows.append(
            (
                path.name,
                line.get("line_num"),
                line.get("lang"),
                line.get("quality_score"),
                line.get("category"),
                line.get("text", ""),
            )
        )
    return rows


def print_table(rows: list[tuple], as_csv: bool) -> None:
    header = ("FILE", "LINE", "LANG", "QUALITY", "CATEGORY", "TEXT")
    if as_csv:
        writer = csv.writer(sys.stdout)
        writer.writerow(header)
        for row in rows:
            quality = "" if row[3] is None else f"{row[3]:.4f}"
            writer.writerow([row[0], row[1], row[2], quality, row[4], row[5]])
    else:
        print(f"{header[0]:<28} {header[1]:>4} {header[2]:<5} {header[3]:>7} {header[4]:<9} {header[5]}")
        for row in rows:
            quality = "      -" if row[3] is None else f"{row[3]:>7.4f}"
            text = row[5] if len(row[5]) <= 50 else row[5][:47] + "..."
            print(f"{row[0]:<28} {row[1]:>4} {str(row[2]):<5} {quality} {str(row[4]):<9} {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="ALTO XML page(s) and/or plain-text file(s) to classify")
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help=f"API base URL (default: {DEFAULT_BASE_URL}, env: ATRIUM_AP_URL)"
    )
    parser.add_argument(
        "--task-type",
        choices=["auto", "alto", "text"],
        default="auto",
        help="input handling: alto (XML layout+classify), text (line classify), auto by suffix (default)",
    )
    parser.add_argument(
        "--format", choices=["table", "csv", "json"], default="table", help="output format (default: table)"
    )
    parser.add_argument("--info", action="store_true", help="print service capabilities and limits, then exit")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    if args.info:
        print(json.dumps(http_json(f"{base_url}/info", timeout=60), indent=2))
        return

    if not args.files:
        parser.error("no input files given (or use --info)")

    paths = [Path(f) for f in args.files]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print(f"File(s) not found: {', '.join(str(p) for p in missing)}", file=sys.stderr)
        sys.exit(1)

    raw_results = {}
    rows = []
    for path in paths:
        result = process_file(base_url, path, task_type=args.task_type)
        if result:
            raw_results[path.name] = result
            rows.extend(result_rows(path, result))

    if not rows:
        print("No results produced.", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        print(json.dumps(raw_results, indent=2, ensure_ascii=False))
    else:
        print_table(rows, as_csv=(args.format == "csv"))


if __name__ == "__main__":
    main()
