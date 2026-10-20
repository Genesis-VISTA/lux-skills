#!/usr/bin/env bash
#
# unpack.sh — flatten lux skills up to an agent's skills directory.
#
# Agent clients commonly discover skills as *direct children* of a configured
# skills directory (without recursion). This repo keeps skills nested by domain
# (skills/<domain>/<skill>/SKILL.md) for browsing, so a plain copy is invisible
# to native discovery. unpack.sh bridges the gap: it flattens each leaf skill
# into <dir>/<skill-name>/ for direct client discovery.
#
# Usage:
#   ./unpack.sh [DOMAIN ...] [options]
#
#   With no DOMAIN args, every domain is unpacked. Pass one or more domain
#   folder names to narrow the set. A flat skill (skills/<skill>/SKILL.md),
#   such as vllm-skill, is selected by its name.
#
# Options:
#   --root PATH          Workdir root that holds .claude/skills and/or
#                        .agents/skills. Default: nearest enclosing .claude or
#                        .agents ancestor of this clone, else $PWD.
#   --target PATH        Literal destination skills dir. Overrides --root and
#                        --harness; skills are written as direct children here.
#   --mode {symlink|copy}  symlink (default; stays in sync) or copy (standalone).
#   --harness {claude|agents|all}  Which dir(s) to write under --root:
#                        .claude/skills (claude, default), .agents/skills
#                        (agents; Codex, Cursor, OpenCode, and Gemini CLI),
#                        or both (all).
#   --list               Print discoverable domains + leaf skills and exit.
#   -h, --help           Show this help.
#
set -euo pipefail
unset CDPATH

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SOURCE_ROOT="$SCRIPT_DIR/skills"

MODE="symlink"
HARNESS="claude"
ROOT=""
TARGET=""
LIST_ONLY=0
DOMAINS=()

require_value() {
  [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 requires a value"
}

die() { printf 'unpack.sh: %s\n' "$1" >&2; exit 1; }
usage() {
  awk '
    NR == 1 { next }
    /^#/ { sub(/^# ?/, ""); print; next }
    { exit }
  ' "${BASH_SOURCE[0]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) require_value "$@"; ROOT="$2"; shift 2 ;;
    --root=*) ROOT="${1#*=}"; [[ -n "$ROOT" ]] || die "--root requires a value"; shift ;;
    --target) require_value "$@"; TARGET="$2"; shift 2 ;;
    --target=*) TARGET="${1#*=}"; [[ -n "$TARGET" ]] || die "--target requires a value"; shift ;;
    --mode) require_value "$@"; MODE="$2"; shift 2 ;;
    --mode=*) MODE="${1#*=}"; [[ -n "$MODE" ]] || die "--mode requires a value"; shift ;;
    --harness) require_value "$@"; HARNESS="$2"; shift 2 ;;
    --harness=*) HARNESS="${1#*=}"; [[ -n "$HARNESS" ]] || die "--harness requires a value"; shift ;;
    --list)      LIST_ONLY=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    -*)          die "unknown option: $1" ;;
    *)           DOMAINS+=("$1"); shift ;;
  esac
done

[[ "$MODE" == "symlink" || "$MODE" == "copy" ]] || die "--mode must be symlink or copy"
case "$HARNESS" in claude|agents|all) ;; *) die "--harness must be claude, agents, or all" ;; esac
[[ -d "$SOURCE_ROOT" ]] || die "skills/ not found next to unpack.sh ($SOURCE_ROOT)"

# Reject a misspelled selection even when another requested domain is valid.
if [[ ${#DOMAINS[@]} -gt 0 ]]; then
  for domain in "${DOMAINS[@]}"; do
    [[ "$domain" != .* && "$domain" != */* && -d "$SOURCE_ROOT/$domain" ]] || \
      die "unknown domain or flat skill: $domain (check --list)"
  done
fi

_wanted() {
  local d="$1" want
  for want in "${DOMAINS[@]}"; do [[ "$want" == "$d" ]] && return 0; done
  return 1
}

# Recurse a domain dir; stop descending once a SKILL.md is found. Skip hidden.
_walk_leaves() {
  local domain="$1" dir="$2" child name
  for child in "$dir"/*/; do
    [[ -d "$child" ]] || continue
    # Catalog directories are real directories. Following arbitrary links here
    # could recurse forever or install files outside the clone.
    [[ ! -L "${child%/}" ]] || die "symlinked catalog directory: ${child%/}"
    name="$(basename "$child")"
    [[ "$name" == .* ]] && continue
    if [[ -f "$child/SKILL.md" ]]; then
      printf '%s\t%s\t%s\n' "$domain" "$name" "${child%/}"
    else
      _walk_leaves "$domain" "${child%/}"
    fi
  done
}

# Emit "domain<TAB>skill_name<TAB>abs_skill_dir" per leaf (mirrors catalog.py:
# stop at first SKILL.md; skip hidden dirs like .curated).
discover() {
  local filter=0; [[ ${#DOMAINS[@]} -gt 0 ]] && filter=1
  local entry domain
  for entry in "$SOURCE_ROOT"/*/; do
    [[ -d "$entry" ]] || continue
    [[ ! -L "${entry%/}" ]] || die "symlinked catalog directory: ${entry%/}"
    domain="$(basename "$entry")"
    [[ "$domain" == .* ]] && continue
    if [[ -f "$entry/SKILL.md" ]]; then
      if [[ $filter -eq 0 ]] || _wanted "$domain"; then
        printf '%s\t%s\t%s\n' "(flat)" "$domain" "${entry%/}"
      fi
      continue
    fi
    if [[ $filter -eq 1 ]] && ! _wanted "$domain"; then continue; fi
    _walk_leaves "$domain" "${entry%/}"
  done
}

# Workdir root: nearest .claude/.agents ancestor of the clone, else $PWD.
default_root() {
  local d="$SCRIPT_DIR" base
  while [[ "$d" != "/" ]]; do
    base="$(basename "$d")"
    if [[ "$base" == ".claude" || "$base" == ".agents" ]]; then
      dirname "$d"; return 0
    fi
    d="$(dirname "$d")"
  done
  printf '%s\n' "$PWD"
}

# Resolve existing ancestor symlinks without requiring GNU realpath or Python.
# Also normalize . and .. before comparing destination/source paths.
canonical_dir() {
  local path="$1" parent base
  if [[ -d "$path" ]]; then
    (cd -P -- "$path" && pwd -P)
  else
    [[ ! -e "$path" && ! -L "$path" ]] || die "not a directory: $path"
    parent="$(canonical_dir "$(dirname -- "$path")")" || exit 1
    base="$(basename -- "$path")"
    case "$base" in
      .) printf '%s\n' "$parent" ;;
      ..) dirname -- "$parent" ;;
      *) printf '%s/%s\n' "${parent%/}" "$base" ;;
    esac
  fi
}

main() {
  local rows; rows="$(discover)"
  [[ -n "$rows" ]] || die "no skills matched (check domain names with --list)"

  local dups; dups="$(printf '%s\n' "$rows" | cut -f2 | sort | uniq -d || true)"
  [[ -z "$dups" ]] || die "duplicate skill name(s) across domains: $(echo "$dups" | tr '\n' ' ')"

  local total; total="$(printf '%s\n' "$rows" | wc -l | tr -d ' ')"

  if [[ $LIST_ONLY -eq 1 ]]; then
    printf '%s\n' "$rows" | awk -F'\t' \
      '{ if ($1=="(flat)") printf "  %-34s (flat)\n", $2; else printf "  %-34s [%s]\n", $2, $1 }' \
      | sort
    printf '\n%s skill(s).\n' "$total"
    return 0
  fi

  # Resolve destination skills dir(s).
  local -a dests=()
  if [[ -n "$TARGET" ]]; then
    dests=("$TARGET")
  else
    [[ -n "$ROOT" ]] || ROOT="$(default_root)"
    case "$HARNESS" in
      claude) dests=("$ROOT/.claude/skills") ;;
      agents) dests=("$ROOT/.agents/skills") ;;
      all)    dests=("$ROOT/.claude/skills" "$ROOT/.agents/skills") ;;
    esac
  fi

  local dest name dir source_path
  source_path="$(cd -- "$SOURCE_ROOT" && pwd -P)"
  # Validate every destination before replacing any existing entry. In
  # particular --target skills/<domain> must never delete the catalog itself.
  local -a resolved_dests=()
  for dest in "${dests[@]}"; do
    dest="$(canonical_dir "$dest")"
    [[ "$dest" != "$source_path" && "$dest" != "$source_path/"* ]] || \
      die "destination overlaps source catalog: $dest"
    while IFS=$'\t' read -r _domain name dir; do
      [[ "$source_path" != "$dest/$name" && "$source_path" != "$dest/$name/"* ]] || \
        die "destination entry would replace the source catalog: $dest/$name"
    done <<< "$rows"
    resolved_dests+=("$dest")
  done

  for dest in "${resolved_dests[@]}"; do
    mkdir -p "$dest"
    while IFS=$'\t' read -r _domain name dir; do
      [[ -n "$name" ]] || continue
      local out="$dest/$name"
      [[ -e "$out" || -L "$out" ]] && rm -rf "$out"
      if [[ "$MODE" == "symlink" ]]; then ln -s "$dir" "$out"; else cp -R "$dir" "$out"; fi
    done <<< "$rows"
    printf 'unpacked %s skill(s) (%s) -> %s\n' "$total" "$MODE" "$dest"
  done
}

main
