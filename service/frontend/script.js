/**
 * script.js
 * Frontend logic for ATRIUM Text Processor
 *
 * Quality categories returned by text_util_langID.py:
 *   Clear    – passes all structural and perplexity checks
 *   Noisy    – degraded but recoverable (symbol / uppercase / high PPL)
 *   Trash    – structurally corrupt, not worth downstream processing
 *   Non-text – numeric / separator-only content (dates, page numbers, codes)
 *   Empty    – blank line
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CATEGORY_STYLES = {
    Clear:      "badge-clear",
    Noisy:      "badge-noisy",
    Trash:      "badge-trash",
    "Non-text": "badge-nontext",
    Empty:      "badge-empty",
    Unknown:    "badge-noisy",
};

/** Categories considered "usable" for the quality-ratio numerator. */
const USABLE_CATEGORIES = new Set(["Clear", "Noisy"]);

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initFileUpload();
    initFormHandler();
});

// ---------------------------------------------------------------------------
// Tab navigation
// ---------------------------------------------------------------------------

function initTabs() {
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(btn.dataset.tab).classList.add("active");
        });
    });
}

// ---------------------------------------------------------------------------
// File input
// ---------------------------------------------------------------------------

function initFileUpload() {
    const input   = document.getElementById("fileInput");
    const display = document.getElementById("fileNameDisplay");
    const box     = document.getElementById("uploadBox");

    box.addEventListener("click", () => input.click());

    box.addEventListener("dragover",  e => { e.preventDefault(); box.classList.add("drag-over"); });
    box.addEventListener("dragleave", ()  => box.classList.remove("drag-over"));
    box.addEventListener("drop", e => {
        e.preventDefault();
        box.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            input.files = e.dataTransfer.files;
            display.textContent = "Selected: " + e.dataTransfer.files[0].name;
        }
    });

    input.addEventListener("change", () => {
        display.textContent = input.files.length > 0
            ? "Selected: " + input.files[0].name
            : "";
    });
}

// ---------------------------------------------------------------------------
// Form submission
// ---------------------------------------------------------------------------

function initFormHandler() {
    document.getElementById("processForm").addEventListener("submit", async e => {
        e.preventDefault();

        const input = document.getElementById("fileInput");
        if (!input.files.length) { alert("Please select a file."); return; }

        const port    = window.location.port;
        const apiBase = (port === "8080" || port === "5500")
            ? "http://localhost:8000"
            : window.location.origin;

        document.getElementById("results").innerHTML         = "";
        document.getElementById("loading").style.display    = "block";
        document.querySelector(".btn-primary").disabled      = true;

        const formData = new FormData(document.getElementById("processForm"));

        try {
            const response = await fetch(`${apiBase}/process`, {
                method: "POST",
                body:   formData,
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || `HTTP ${response.status}`);
            }

            renderResults(await response.json());

        } catch (err) {
            document.getElementById("results").innerHTML = `
                <div class="error-box">
                    <strong>Error:</strong> ${escapeHtml(err.message)}
                </div>`;
        } finally {
            document.getElementById("loading").style.display = "none";
            document.querySelector(".btn-primary").disabled  = false;
        }
    });
}

// ---------------------------------------------------------------------------
// Results rendering
// ---------------------------------------------------------------------------

function renderResults(data) {
    const container = document.getElementById("results");
    const lines     = data.cleaned_lines ?? [];

    // --- Summary bar ---
    const total   = lines.length;
    const usable  = lines.filter(l => USABLE_CATEGORIES.has(l.category)).length;
    const cleared = lines.filter(l => l.category === "Clear").length;
    const noisy   = lines.filter(l => l.category === "Noisy").length;
    const trash   = lines.filter(l => l.category === "Trash").length;
    const nontext = lines.filter(l => l.category === "Non-text").length;
    const empty   = lines.filter(l => l.category === "Empty").length;

    // Page-level averages for word_weird and quality_score
    // (only lines that went through GPU scoring contribute, matching
    //  the avg_quality_score / avg_word_weird logic in langID_aggregate)
    const scoredLines = lines.filter(l =>
        l.category !== "Empty" && l.category !== "Non-text"
    );
    const avgWeird = scoredLines.length
        ? (scoredLines.reduce((s, l) => s + (l.word_weird ?? 0), 0) / scoredLines.length)
        : null;
    const avgQuality = scoredLines.length
        ? (scoredLines.reduce((s, l) => s + (l.quality_score ?? 0), 0) / scoredLines.length)
        : null;

    let html = `
        <div class="summary-stats">
            <div>
                <span class="stat-label">File</span>
                <span class="stat-value">${escapeHtml(data.filename ?? "")}</span>
            </div>
            <div>
                <span class="stat-label">Type</span>
                <span class="stat-value">${escapeHtml(data.type ?? "")}</span>
            </div>
            <div>
                <span class="stat-label">Lines</span>
                <span class="stat-value">${total}</span>
            </div>
            <div>
                <span class="stat-label">Usable</span>
                <span class="stat-value">${usable}/${total}</span>
            </div>
            ${avgQuality !== null ? `
            <div>
                <span class="stat-label">Avg Quality</span>
                <span class="stat-value">${avgQuality.toFixed(3)}</span>
            </div>
            <div>
                <span class="stat-label">Avg Weirdness</span>
                <span class="stat-value">${avgWeird.toFixed(3)}</span>
            </div>` : ""}
        </div>
        <div class="category-breakdown">
            <span class="badge badge-clear">Clear ${cleared}</span>
            <span class="badge badge-noisy">Noisy ${noisy}</span>
            <span class="badge badge-trash">Trash ${trash}</span>
            <span class="badge badge-nontext">Non-text ${nontext}</span>
            <span class="badge badge-empty">Empty ${empty}</span>
        </div>`;

    // --- Line table ---
    if (lines.length > 0) {
        html += `
        <h3>Line Analysis</h3>
        <div class="data-table-wrapper">
            <table class="data-table">
                <thead>
                    <tr>
                        <th style="width:44px">#</th>
                        <th>Text</th>
                        <th style="width:50px"
                            title="Tokens containing characters outside the allowed set (detect_strange_symbols)">
                            Sym</th>
                        <th style="width:56px"
                            title="Tokens with mid-word uppercase artefacts (detect_mid_uppercase)">
                            Upper</th>
                        <th style="width:80px"
                            title="DistilGPT2 perplexity; unreliable on short lines or non-English text">
                            PPL</th>
                        <th style="width:76px"
                            title="Mean per-word weirdness [0–1]: combines strange-symbol, repeated-symbol, LDL-fusion and mid-uppercase signals. 0 = fully clean.">
                            Weird</th>
                        <th style="width:76px"
                            title="Composite quality score [0–1]: aggregates valid-word ratio, symbol ratio, perplexity and text length. Higher = cleaner.">
                            Quality</th>
                        <th style="width:90px">Status</th>
                    </tr>
                </thead>
                <tbody>`;

        lines.forEach(row => {
            const badgeClass   = CATEGORY_STYLES[row.category] ?? CATEGORY_STYLES.Unknown;
            const pplDisplay   = (row.perplexity != null && row.perplexity > 0)
                ? row.perplexity.toFixed(1) : "—";
            const symDisplay   = row.sym_count   != null ? row.sym_count   : "—";
            const upperDisplay = row.upper_count != null ? row.upper_count : "—";

            // word_weird displayed as percentage for readability
            const weirdDisplay = row.word_weird != null
                ? (row.word_weird * 100).toFixed(1) + "%" : "—";

            // quality_score colour hint: green > 0.75, amber ≥ 0.45, red < 0.45
            let qClass = "";
            if (row.quality_score != null) {
                qClass = row.quality_score > 0.75 ? "q-high"
                       : row.quality_score >= 0.45 ? "q-mid"
                       : "q-low";
            }
            const qualityDisplay = row.quality_score != null
                ? row.quality_score.toFixed(3) : "—";

            html += `
                    <tr class="row-${(row.category ?? "").toLowerCase().replace("-", "")}">
                        <td><small>${row.line_num}</small></td>
                        <td class="text-cell">${escapeHtml(row.text)}</td>
                        <td class="num-cell">${symDisplay}</td>
                        <td class="num-cell">${upperDisplay}</td>
                        <td class="num-cell">${pplDisplay}</td>
                        <td class="num-cell">${weirdDisplay}</td>
                        <td class="num-cell ${qClass}">${qualityDisplay}</td>
                        <td><span class="badge ${badgeClass}">${escapeHtml(row.category)}</span></td>
                    </tr>`;
        });

        html += `       </tbody>
            </table>
        </div>`;
    } else {
        html += `<p class="empty-msg">No text lines extracted.</p>`;
    }

    // --- Raw text toggle ---
    if (data.raw_text) {
        html += `
        <div class="raw-section">
            <button type="button" class="btn-secondary" onclick="toggleRaw(this)">
                Show Raw Extracted Text
            </button>
            <pre id="rawTextView" class="raw-text-box" hidden>${escapeHtml(data.raw_text)}</pre>
        </div>`;
    }

    container.innerHTML = html;
}

function toggleRaw(btn) {
    const pre = document.getElementById("rawTextView");
    const hidden = pre.hidden;
    pre.hidden = !hidden;
    btn.textContent = hidden ? "Hide Raw Extracted Text" : "Show Raw Extracted Text";
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function escapeHtml(text) {
    if (text == null) return "";
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
