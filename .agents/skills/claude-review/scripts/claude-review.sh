#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-}"
CONTEXT_LINES="${CLAUDE_REVIEW_CONTEXT_LINES:-40}"
TIMEOUT_SECONDS="${CLAUDE_REVIEW_TIMEOUT_SECONDS:-300}"
DIFF_CMD=()

PROMPT='You are an independent senior code reviewer. Review the git diff from stdin.
Focus only on high-confidence correctness, security, concurrency, data-loss, API-contract, and regression risks.
Do not comment on style unless it creates a real bug.
Do not edit files.
If the diff is empty, say that there are no changes to review.
Output:
- Summary
- Critical findings
- Major findings
- Minor findings
- Suggested tests
Include file paths and line hints when possible.'

if ! command -v claude >/dev/null 2>&1; then
  echo "claude CLI not found on PATH." >&2
  exit 127
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "claude-review must be run from inside a Git worktree." >&2
  exit 2
fi

if ! [[ "$CONTEXT_LINES" =~ ^[1-9][0-9]*$ ]]; then
  echo "CLAUDE_REVIEW_CONTEXT_LINES must be a positive integer." >&2
  exit 2
fi

if ! [[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "CLAUDE_REVIEW_TIMEOUT_SECONDS must be a non-negative integer." >&2
  exit 2
fi

run_claude() {
  local input_file timeout_file pid watcher status

  input_file="$(mktemp "${TMPDIR:-/tmp}/claude-review.XXXXXX")"
  timeout_file="$(mktemp "${TMPDIR:-/tmp}/claude-review-timeout.XXXXXX")"
  rm -f "$timeout_file"

  cleanup_claude_review() {
    trap - RETURN

    if [[ -n "${watcher:-}" ]]; then
      kill "$watcher" >/dev/null 2>&1 || true
      wait "$watcher" 2>/dev/null || true
    fi

    if [[ -n "${pid:-}" ]]; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" 2>/dev/null || true
    fi

    rm -f "$input_file" "$timeout_file"
  }

  trap cleanup_claude_review RETURN

  cat >"$input_file"

  claude -p \
    --no-session-persistence \
    --output-format text \
    --permission-mode plan \
    --tools=Read,Grep,Glob \
    "$PROMPT" <"$input_file" &
  pid=$!

  if ((TIMEOUT_SECONDS > 0)); then
    (
      sleep "$TIMEOUT_SECONDS"
      if kill -0 "$pid" >/dev/null 2>&1; then
        : >"$timeout_file"
        kill "$pid" >/dev/null 2>&1 || true
      fi
    ) &
    watcher=$!
  else
    watcher=""
  fi

  if wait "$pid"; then
    status=0
  else
    status=$?
  fi

  if [[ -n "$watcher" ]]; then
    kill "$watcher" >/dev/null 2>&1 || true
    wait "$watcher" 2>/dev/null || true

    if [[ -e "$timeout_file" ]]; then
      echo "claude-review timed out after ${TIMEOUT_SECONDS}s." >&2
      status=124
    fi
  fi

  if ((status != 0)) && [[ ! -e "$timeout_file" ]]; then
    echo "claude CLI exited with status ${status}; review aborted." >&2
  fi

  return "$status"
}

if [[ -n "$BASE" ]]; then
  if ! git rev-parse --verify --quiet "$BASE^{commit}" >/dev/null; then
    echo "Unknown base ref: $BASE" >&2
    exit 2
  fi

  DIFF_CMD=(git diff --unified="$CONTEXT_LINES" "$BASE...HEAD" --)
elif git rev-parse --verify HEAD >/dev/null 2>&1; then
  DIFF_CMD=(git diff --unified="$CONTEXT_LINES" HEAD --)
else
  DIFF_CMD=(git diff --unified="$CONTEXT_LINES" --)
fi

"${DIFF_CMD[@]}" | run_claude
