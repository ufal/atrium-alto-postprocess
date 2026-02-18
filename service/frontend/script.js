/**
 * script.js
 * Frontend logic for Atrium Text Processor
 */

// --- Constants ---
const CATEGORY_STYLES = {
    "Clear": "badge-clear",
    "Noisy": "badge-noisy",
    "Trash": "badge-trash",
    "Unknown": "badge-noisy"
};

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
        const tabId = $(this).data('tab');
        $('#' + tabId).addClass('active');
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

        // UI State: Loading
        $('#results').empty();
        $('#loading').show();
        $('.btn-primary').prop('disabled', true);

        const formData = new FormData(this);

        try {
            // Determine API URL (Dev vs Prod)
            let baseUrl = window.location.origin;
            // If serving frontend via webpack/live-server on 8080/5500, point to Python 8000
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

            const data = await response.json();
            renderResults(data);

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

    // 1. Header & Stats
    const totalLines = data.cleaned_lines ? data.cleaned_lines.length : 0;
    const cleanCount = data.cleaned_lines ? data.cleaned_lines.filter(l => l.category === "Clear").length : 0;

    let html = `
        <div class="summary-stats">
            <div><span class="stat-item">File:</span> <span class="stat-value">${data.filename}</span></div>
            <div><span class="stat-item">Type:</span> <span class="stat-value">${data.type}</span></div>
            <div><span class="stat-item">Total Lines:</span> <span class="stat-value">${totalLines}</span></div>
            <div><span class="stat-item">Quality Ratio:</span> <span class="stat-value">${cleanCount}/${totalLines}</span></div>
        </div>
    `;

    // 2. Cleaned Lines Table
    if (data.cleaned_lines && data.cleaned_lines.length > 0) {
        html += `<h3>Line Analysis</h3>`;
        html += `<div class="data-table-wrapper"><table class="data-table">`;
        html += `<thead><tr>
                    <th style="width: 50px">#</th>
                    <th>Text Content</th>
                    <th style="width: 80px">Lang</th>
                    <th style="width: 80px">PPL</th>
                    <th style="width: 100px">Status</th>
                 </tr></thead><tbody>`;

        data.cleaned_lines.forEach(row => {
            const badgeClass = CATEGORY_STYLES[row.category] || "badge-noisy";
            // Highlight splits if they exist
            let displayText = row.text;

            html += `<tr>
                <td><small class="text-muted">${row.line_num}</small></td>
                <td>${displayText}</td>
                <td><code>${row.lang}</code> (${row.lang_conf})</td>
                <td>${row.perplexity}</td>
                <td><span class="badge ${badgeClass}">${row.category}</span></td>
            </tr>`;
        });

        html += `</tbody></table></div>`;
    } else {
        html += `<p class="text-muted">No text lines extracted.</p>`;
    }

    // 3. Raw Text (Collapsible)
    if (data.raw_text) {
        html += `
            <div style="margin-top: 2rem;">
                <button type="button" class="btn-primary" style="background:#6c757d; margin:0;"
                        onclick="$('#rawTextView').toggle()">
                    Toggle Raw Extracted Text
                </button>
                <div id="rawTextView" style="display:none;">
                    <div class="raw-text-box">${escapeHtml(data.raw_text)}</div>
                </div>
            </div>
        `;
    }

    container.html(html);
}

function escapeHtml(text) {
    if (!text) return "";
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}