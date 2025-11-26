import sys
import re
import os

def extract_changelog(version, changelog_path='CHANGELOG.md'):
    """
    Extracts the changelog section for a specific version.

    Accepts headers like:
      ## [v0.1.3] - 2025-11-27
      ## v0.1.3
      ## [0.1.3]
      ## 0.1.3 - yyyy-mm-dd
    """
    # Normalize whitespace; keep original for fallback attempts
    original_version = version.strip()

    if not os.path.exists(changelog_path):
        print(f"Error: {changelog_path} not found.", file=sys.stderr)
        return None

    try:
        with open(changelog_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {changelog_path}: {e}", file=sys.stderr)
        return None

    # prepare candidate version forms to try (with/without leading 'v')
    candidates = []
    v_stripped = original_version.lstrip('vV')
    candidates.append(v_stripped)
    # if user gave version without v, also try with v
    if not original_version.lower().startswith('v'):
        candidates.append('v' + v_stripped)
    # if user gave version with v, also try without
    if original_version.lower().startswith('v') and v_stripped not in candidates:
        candidates.append(v_stripped)

    # Build a flexible regex that allows:
    #  - optional brackets [ ... ] around version
    #  - optional whitespace between ## and version
    #  - optional 'v' prefix (we already include candidates, but keep pattern tolerant)
    #  - consume header line (which may include date) then capture until next "## " or EOF
    pattern_template = r'^##\s*\[?\s*{ver}\s*\]?.*?\n(.*?)(?=^##\s|\Z)'

    for cand in candidates:
        pat = pattern_template.format(ver=re.escape(cand))
        pattern = re.compile(pat, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        match = pattern.search(content)
        if match:
            return match.group(1).strip()

    # as a last resort, try a looser search: header that contains the version anywhere on the header line
    loose_pattern = re.compile(
        r'^##.*' + re.escape(v_stripped) + r'.*?\n(.*?)(?=^##\s|\Z)',
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    loose_match = loose_pattern.search(content)
    if loose_match:
        return loose_match.group(1).strip()

    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_changelog.py <version> [changelog_path]", file=sys.stderr)
        sys.exit(1)

    version = sys.argv[1]
    changelog_path = sys.argv[2] if len(sys.argv) > 2 else 'CHANGELOG.md'

    content = extract_changelog(version, changelog_path)

    if content:
        print(content)
    else:
        # no output (workflow can detect empty)
        pass
