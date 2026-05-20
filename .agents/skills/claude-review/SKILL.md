---
name: claude-review
description: Use the local Claude Code CLI as an independent read-only reviewer for the current git diff or a branch diff. Trigger when the user asks Codex to ask Claude, use Claude Code, get a Claude second opinion, or run an adversarial code review.
---

# Claude Review

Use the local Claude Code CLI as an independent read-only reviewer. Claude is a second opinion, not ground truth.

## Workflow

1. Verify `claude` is available with `claude auth status` or `claude -v`.
2. From the target Git worktree, run this skill's `scripts/claude-review.sh`.
3. Pass an optional base ref as the first argument to review `BASE...HEAD`; omit it to review current tracked working-tree changes against `HEAD`.
4. The branch form uses three-dot diff semantics, so it reviews changes since the merge base with `BASE`.
5. For new untracked files, run `git add -N <path>` first so they appear in the diff without staging content.
6. Do not ask Claude to edit files.
7. Summarize Claude's findings for the user as:
   - severity
   - file / line if available
   - issue
   - why it matters
   - suggested fix

This skill never writes files; it produces a review report only. Codex should implement fixes only when the user asks.
