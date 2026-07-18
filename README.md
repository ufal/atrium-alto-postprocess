<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" title="Python Version"></a>
  <a href="https://huggingface.co/facebook/fasttext-language-identification"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-fasttext--langID-yellow.svg" title="FastText Language Identification"></a>
  <a href="https://huggingface.co/Qwen/Qwen2.5-0.5B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Qwen2.5--0.5B-yellow.svg" title="Qwen2.5-0.5B Perplexity"></a>
  <a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/github/license/ufal/atrium-alto-postprocess" title="MIT License"></a>
  <a href="https://atrium-research.eu/"><img src="https://img.shields.io/badge/funded%20by-ATRIUM-8A2BE2.svg" title="ATRIUM Project"></a>
</p>

---

# ATRIUM ALTO Postprocessing - Agent Skill 🤖🧹

### Goal: let coding agents classify OCR line quality via a server-client skill

This branch (`agent-skill`) packages the **ATRIUM ALTO Postprocessing API service**
together with a **Skill for coding agents** (Claude Code, Codex, Gemini/Antigravity).
The design follows a strict server-client split:

- **Server** 🖥️ - the FastAPI service in [`service/`](service/) runs LayoutReader,
  FastText language ID, and Qwen2.5-0.5B perplexity (Docker Compose `api` profile
  or local venv, CPU or GPU).
- **Client** 🪶 - [`scripts/atrium_postprocess.py`](scripts/atrium_postprocess.py),
  a **zero-dependency** stdlib-only script that agents call directly.
- **Skill contract** 📜 - [`SKILL.md`](SKILL.md) tells the agent when and how to use
  it: quality-category semantics, routing rules, error playbooks.

For the batch pipeline, statistics tooling, and full project documentation, see the
[`test`](https://github.com/ufal/atrium-alto-postprocess/tree/test) branch - this
branch intentionally carries only what the skill needs.

### Table of contents 📑

  * [Quick start 🚀](#quick-start-)
  * [Skill installation 🔧](#skill-installation-)
  * [Server setup 🖥️](#server-setup-)
  * [Client usage 🪶](#client-usage-)
  * [Remote server / LINDAT 🌐](#remote-server--lindat-)
  * [Maintenance notes 🔍](#maintenance-notes-)
  * [Contacts 📧](#contacts-)

----

## Quick start 🚀

```bash
git clone -b agent-skill https://github.com/ufal/atrium-alto-postprocess.git
cd atrium-alto-postprocess

bash scripts/server.sh                                                      # start the server
python3 scripts/atrium_postprocess.py small_data_samples/CTX000000001-1.alto.xml
```

> [!NOTE]
> The first server start downloads FastText `lid.176.bin` (~130 MB), LayoutReader,
> and the Qwen2.5-0.5B perplexity model - be patient. ⏳

## Skill installation 🔧

### Claude Code

```bash
git clone -b agent-skill https://github.com/ufal/atrium-alto-postprocess.git \
    ~/.claude/skills/atrium-alto-postprocess
```

Restart Claude Code - the skill is available as `/atrium-alto-postprocess` and is
selected automatically for OCR quality-filtering requests. For a project-local
install, clone into `.claude/skills/atrium-alto-postprocess` inside the target
repository.

### Codex

```bash
git clone -b agent-skill https://github.com/ufal/atrium-alto-postprocess.git \
    ~/.codex/skills/atrium-alto-postprocess
```

The skill is detected automatically in the next Codex session.

### Google Antigravity

Clone the branch into your project and point `AGENTS.md` at it:

```
Use the ATRIUM ALTO postprocessing skill from
`atrium-alto-postprocess/SKILL.md` for classifying OCR line quality.
Start the server with `bash atrium-alto-postprocess/scripts/server.sh`, then run
`python3 atrium-alto-postprocess/scripts/atrium_postprocess.py [FILES...]`.
```

Update any install with `git pull` inside the cloned skill directory.

## Server setup 🖥️

The server exposes three endpoints (see [`service/README.md`](service/README.md)
for details): `GET /info`, `GET /health`, `POST /process`.

```bash
bash scripts/server.sh          # auto: Docker Compose api profile, else local uvicorn
bash scripts/server.sh --gpu    # Docker with GPU overlay
bash scripts/server.sh --local  # force local uvicorn via setup/setup_api_server.sh
```

The script is idempotent and health-waits on `/info`. Port defaults to `8000`
(`ATRIUM_AP_PORT` to change).

## Client usage 🪶

```bash
python3 scripts/atrium_postprocess.py page.alto.xml              # ALTO page
python3 scripts/atrium_postprocess.py page.txt --format csv      # text lines → CSV
python3 scripts/atrium_postprocess.py scans/*.xml --format json  # full metrics
python3 scripts/atrium_postprocess.py --info                     # capabilities
```

Output rows: `FILE, LINE, LANG, QUALITY, CATEGORY, TEXT`. The five quality
categories 🧹 and their routing semantics are documented in
[`SKILL.md`](SKILL.md#quality-categories-).

## Remote server / LINDAT 🌐

The client is location-agnostic: point it at any deployment with `--base-url` or

```bash
export ATRIUM_AP_URL="https://<hosted-instance>/atrium-ap"
```

A hosted LINDAT instance is planned; once available, the environment variable is the
only change needed - the skill contract and client stay identical.

## Maintenance notes 🔍

Review checklist for every change / sync-merge into this branch (the ATRIUM skill
anti-pattern checklist):

- [ ] no doc references a script name that differs from the committed file;
- [ ] no provenance/paradata claim unless the service imports it on this branch;
- [ ] no reference to directories/files absent from this branch;
- [ ] documented response fields match what `service/text_api.py` actually returns;
- [ ] client smoke test re-run on `small_data_samples/` against a locally started server.

## Contacts 📧

**For support write to:** lutsai.k@gmail.com responsible for the
[GitHub repository](https://github.com/ufal/atrium-alto-postprocess)

### Acknowledgements 🙏

- **Developed by** UFAL, Charles University 👥
- **Funded by** [ATRIUM](https://atrium-research.eu/) 💰
- **Models**: FastText langID, LayoutLMv3 LayoutReader, Qwen2.5-0.5B perplexity 🔗
