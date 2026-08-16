# Wiki source

These files are the **source of truth** for the [GitHub wiki](https://github.com/krishddd/weighted-emergent-bias/wiki).

GitHub wikis live in a separate git repository (`<repo>.wiki.git`) that is not version-controlled
alongside the code and has no review process. Keeping the pages here means wiki edits go through
the same history and review as everything else; the wiki repo is treated as a publish target.

## Publishing

The wiki repo only exists once the wiki has been initialised — create any page once via
**Settings → Features → Wikis**, then the first page in the web UI. After that:

```bash
git clone https://github.com/krishddd/weighted-emergent-bias.wiki.git /tmp/web-wiki
cp wiki/*.md /tmp/web-wiki/
cd /tmp/web-wiki && git add -A && git commit -m "Sync wiki from main" && git push
```

## Page naming

GitHub maps a filename to a page title by replacing hyphens with spaces, so `Getting-Started.md`
becomes "Getting Started". Inter-page links use the filename without extension —
`[Invariants](Invariants)`, not a relative path. `_Sidebar.md` renders as the sidebar on every
page.

Mermaid fences render natively in the wiki; no plugin is required.
