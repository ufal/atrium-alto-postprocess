/**
 * script.js
 * Frontend logic for Atrium Text Processor (LINDAT variant)
 */

// --- Constants ---
const CATEGORY_STYLES = {
    "Clear":    "badge-clear",
    "Noisy":    "badge-noisy",
    "Trash":    "badge-trash",
    "Non-text": "badge-nontext",
    "Empty":    "badge-empty",
    "Unknown":  "badge-noisy",
};

const USABLE_CATEGORIES = new Set(["Clear", "Noisy"]);

// --- Main Init ---
$(document).ready(function() {
    initTabs();
    initFileUpload();
    initFormHandler();
});

// --- Tab Logic ---
function initTabs() {
    $('.tab-btn').click(function() {
        $('.tab-btn').removeClass('active');
        $('.tab-content').removeClass('active');
        $(this).addClass('active');
        $('#' + $(this).data('tab')).addClass('active');
    });
}

// --- File Input Logic ---
function initFileUpload() {
    $('#fileInput').change(function(e) {
        if (e.target.files.length > 0) {
            $('#fileNameDisplay').text("Selected: " + e.target.files[0].name);
        } else {
            $('#fileNameDisplay').text("");
        }
    });
}

// --- Form Submission ---
function initFormHandler() {
    $('#processForm').submit(async function(e) {
        e.preventDefault();

        const fileInput = document.getElementById('fileInput');
        if (fileInput.files.length === 0) {
            alert("Please select a file.");
            return;
        }

        $('#results').empty();
        $('#loading').show();
        $('.btn-primary').prop('disabled', true);

        const formData = new FormData(this);

        try {
            let baseUrl = window.location.origin;
            if (window.location.port === "8080" || window.location.port === "5500") {
                baseUrl = "http://localhost:8000";
            }

            const response = await fetch(`${baseUrl}/process`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Server Error");
            }

            renderResults(await response.json());

        } catch (error) {
            console.error(error);
            $('#results').html(`
                <div style="background:#f8d7da; color:#721c24; padding:1rem; border-radius:4px;">
                    <strong>Error:</strong> ${error.message}
                </div>
            `);
        } finally {
            $('#loading').hide();
            $('.btn-primary').prop('disabled', false);
        }
    });
}

// --- Rendering Logic ---
function renderResults(data) {
    const container = $('#results');
    const lines     = data.cleaned_lines || [];

    const total   = lines.length;
    const usable  = lines.filter(l => USABLE_CATEGORIES.has(l.category)).length;
    const cleared = lines.filter(l => l.category === "Clear").length;
    const noisy   = lines.filter(l => l.category === "Noisy").length;
    const trash   = lines.filter(l => l.category === "Trash").length;
    const nontext = lines.filter(l => l.category === "Non-text").length;
    const empty   = lines.filter(l => l.category === "Empty").length;

    // Page-level averages — exclude Empty and Non-text (matches batch pipeline)
    const scored    = lines.filter(l => l.category !== "Empty" && l.category !== "Non-text");
    const avgWeird  = scored.length
        ? (scored.reduce((s, l) => s + (l.word_weird || 0), 0) / scored.length)
        : null;
    const avgQuality = scored.length
        ? (scored.reduce((s, l) => s + (l.quality_score || 0), 0) / scored.length)
        : null;

    // Summary
    let html = `
        <div class="summary-stats">
            <div><span class="stat-item">File:</span>
                 <span class="stat-value">${data.filename}</span></div>
            <div><span class="stat-item">Type:</span>
                 <span class="stat-value">${data.type}</span></div>
            <div><span class="stat-item">Total Lines:</span>
                 <span class="stat-value">${total}</span></div>
            <div><span class="stat-item">Usable (Clear+Noisy):</span>
                 <span class="stat-value">${usable}/${total}</span></div>
            ${avgQuality !== null ? `
            <div><span class="stat-item">Avg Quality:</span>
                 <span class="stat-value">${avgQuality.toFixed(3)}</span></div>
            <div><span class="stat-item">Avg Weirdness:</span>
                 <span class="stat-value">${avgWeird.toFixed(3)}</span></div>` : ""}
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:1rem;">
            <span class="badge badge-clear">Clear ${cleared}</span>
            <span class="badge badge-noisy">Noisy ${noisy}</span>
            <span class="badge badge-trash">Trash ${trash}</span>
            <span class="badge badge-nontext" style="background:#ede9fe;color:#4c1d95;">Non-text ${nontext}</span>
            <span style="background:#f1f5f9;color:#475569;padding:3px 9px;border-radius:99px;font-size:.75rem;font-weight:700;">Empty ${empty}</span>
        </div>`;

    // Line table
    if (lines.length > 0) {
        html += `<h3>Line Analysis</h3>`;
        html += `<div class="data-table-wrapper"><table class="data-table">`;
        html += `<thead><tr>
                    <th style="width:44px">#</th>
                    <th>Text Content</th>
                    <th style="width:50px"
                        title="Tokens with strange symbols (detect_strange_symbols)">Sym</th>
                    <th style="width:56px"
                        title="Tokens with mid-word uppercase artefacts (detect_mid_uppercase)">Upper</th>
                    <th style="width:80px"
                        title="DistilGPT2 perplexity">PPL</th>
                    <th style="width:76px"
                        title="Mean per-word weirdness [0–1]: combines strange-symbol, repeated-symbol, LDL-fusion and mid-uppercase signals. 0 = fully clean.">Weird</th>
                    <th style="width:76px"
                        title="Composite quality score [0–1]: aggregates valid-word ratio, symbol ratio, perplexity and text length. Higher = cleaner.">Quality</th>
                    <th style="width:90px">Status</th>
                 </tr></thead><tbody>`;

        lines.forEach(row => {
            const badgeClass   = CATEGORY_STYLES[row.category] || "badge-noisy";
            const pplDisplay   = (row.perplexity != null && row.perplexity > 0)
                ? row.perplexity.toFixed(1) : "—";
            const symDisplay   = row.sym_count   != null ? row.sym_count   : "—";
            const upperDisplay = row.upper_count != null ? row.upper_count : "—";

            const weirdDisplay = row.word_weird != null
                ? (row.word_weird * 100).toFixed(1) + "%" : "—";

            // quality_score colour via inline style
            let qStyle = "";
            if (row.quality_score != null) {
                qStyle = row.quality_score > 0.75 ? "color:#0d5c30;font-weight:600"
                       : row.quality_score >= 0.45 ? "color:#78350f;font-weight:600"
                       : "color:#7f1d1d;font-weight:600";
            }
            const qualityDisplay = row.quality_score != null
                ? row.quality_score.toFixed(3) : "—";

            html += `<tr>
                <td><small class="text-muted">${row.line_num}</small></td>
                <td style="font-family:monospace;font-size:.82rem;word-break:break-word;max-width:400px">${escapeHtml(row.text)}</td>
                <td style="text-align:right;font-family:monospace;font-size:.82rem">${symDisplay}</td>
                <td style="text-align:right;font-family:monospace;font-size:.82rem">${upperDisplay}</td>
                <td style="text-align:right;font-family:monospace;font-size:.82rem">${pplDisplay}</td>
                <td style="text-align:right;font-family:monospace;font-size:.82rem">${weirdDisplay}</td>
                <td style="text-align:right;font-family:monospace;font-size:.82rem;${qStyle}">${qualityDisplay}</td>
                <td><span class="badge ${badgeClass}">${escapeHtml(row.category)}</span></td>
            </tr>`;
        });

        html += `</tbody></table></div>`;
    } else {
        html += `<p class="text-muted">No text lines extracted.</p>`;
    }

    // Raw text toggle
    if (data.raw_text) {
        html += `
            <div style="margin-top:2rem;">
                <button type="button" class="btn-primary"
                        style="background:#6c757d;margin:0;"
                        onclick="$('#rawTextView').toggle()">
                    Toggle Raw Extracted Text
                </button>
                <div id="rawTextView" style="display:none;">
                    <div class="raw-text-box">${escapeHtml(data.raw_text)}</div>
                </div>
            </div>`;
    }

    container.html(html);
}

function escapeHtml(text) {
    if (text == null) return "";
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}