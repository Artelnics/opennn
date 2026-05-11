"""Strip // comments immediately preceding function declarations/definitions.

Catches both:
  - Out-of-line definitions at namespace scope: `void foo(...) { ... }`
  - Class/struct member declarations and inline defs (any indentation):
        `void foo(...);`
        `void foo(...) const = 0;`
        `void foo(...,`   (multi-line first line)
"""
import os
import re

ROOT = r"c:\tests_delete\opennn\opennn"

# Match a line that looks like a function decl/def header:
#   <type-words> <name>(   ...possibly more on this or following lines...
# Allows any leading whitespace (catches indented class members).
# The "two-words-min" `\s+` between the type chunk and the name avoids matching
# function CALLS like `foo(...)` (which only have one word before `(`).
FUNC_RE = re.compile(
    r'^\s*[A-Za-z_][\w:&<>*\s,]+\s+([A-Za-z_]\w*::)?[~A-Za-z_]\w*\s*\('
)

# Even with the "two-words" rule, control-flow keywords leak in (`else if (`,
# `for (`, etc. cannot match because they're one word, but `auto x = func(...)`
# style lines are excluded by the regex's lack of `=`. Still, defensively skip
# lines whose first word is a control-flow keyword.
EXCLUDE_FIRST_WORD = {
    'if', 'else', 'while', 'for', 'switch', 'case', 'default', 'do',
    'return', 'throw', 'catch', 'try', 'sizeof', 'new', 'delete',
}

FIRST_WORD_RE = re.compile(r'^\s*([A-Za-z_]\w*)')


def is_function_header(line: str) -> bool:
    if not FUNC_RE.match(line):
        return False
    m = FIRST_WORD_RE.match(line)
    if m and m.group(1) in EXCLUDE_FIRST_WORD:
        return False
    return True


def strip_pre_function_comments(path: str) -> int:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    lines = text.split('\n')

    to_delete = set()

    for i, line in enumerate(lines):
        if not is_function_header(line):
            continue
        # Walk backwards collecting comment lines (allowing blank lines between).
        j = i - 1
        comment_lines = []
        while j >= 0:
            stripped = lines[j].rstrip()
            if stripped == '':
                j -= 1
                continue
            if stripped.lstrip().startswith('//'):
                comment_lines.append(j)
                j -= 1
                continue
            break
        # Don't delete if there's nothing above (file-top comments).
        if comment_lines and j >= 0:
            for cj in comment_lines:
                to_delete.add(cj)
            # Collapse blank lines between deleted comments and the function.
            k = max(comment_lines) + 1
            while k < i and lines[k].strip() == '':
                to_delete.add(k)
                k += 1
            # And blank lines immediately above the deleted comment block.
            k = min(comment_lines) - 1
            while k > j and lines[k].strip() == '':
                to_delete.add(k)
                k -= 1

    if not to_delete:
        return 0

    new_lines = [l for idx, l in enumerate(lines) if idx not in to_delete]
    new_text = '\n'.join(new_lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_text)
    return len(to_delete)


total = 0
for fname in sorted(os.listdir(ROOT)):
    if not (fname.endswith('.cpp') or fname.endswith('.h') or fname.endswith('.cu') or fname.endswith('.cuh')):
        continue
    n = strip_pre_function_comments(os.path.join(ROOT, fname))
    if n > 0:
        print(f"  {fname}: {n} lines")
        total += n
print(f"\nTotal lines deleted: {total}")
