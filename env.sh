# Load project environment for Talking-Head Full Pipeline Assistant.
# Usage (from the repository root, in bash or zsh):
#   source env.sh
#
# Reads .env when that file exists. Canonical variables are ANTHROPIC_*.
# PARALLIGHT_BASE_URL and PARALLIGHT_API_KEY are legacy aliases: they are
# copied onto ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN only when the
# canonical variable is empty. Existing ANTHROPIC_* values are left alone.
# Secrets are not printed.

_th_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if [[ -f "$_th_root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$_th_root/.env"
  set +a
fi

if [[ -z "${ANTHROPIC_BASE_URL:-}" && -n "${PARALLIGHT_BASE_URL:-}" ]]; then
  export ANTHROPIC_BASE_URL="$PARALLIGHT_BASE_URL"
fi

if [[ -z "${ANTHROPIC_AUTH_TOKEN:-}" && -z "${ANTHROPIC_API_KEY:-}" && -n "${PARALLIGHT_API_KEY:-}" ]]; then
  export ANTHROPIC_AUTH_TOKEN="$PARALLIGHT_API_KEY"
fi

# Homebrew ffmpeg (Apple Silicon, then Intel) when those prefixes exist.
if [[ -d /opt/homebrew/bin ]]; then
  case ":$PATH:" in
    *":/opt/homebrew/bin:"*) ;;
    *) export PATH="/opt/homebrew/bin:$PATH" ;;
  esac
fi
if [[ -d /usr/local/bin ]]; then
  case ":$PATH:" in
    *":/usr/local/bin:"*) ;;
    *) export PATH="/usr/local/bin:$PATH" ;;
  esac
fi

if [[ -x "$_th_root/.venv/bin/python" ]]; then
  case ":$PATH:" in
    *":$_th_root/.venv/bin:"*) ;;
    *) export PATH="$_th_root/.venv/bin:$PATH" ;;
  esac
fi

_th_llm="missing"
if [[ -n "${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
  _th_llm="ANTHROPIC_AUTH_TOKEN"
elif [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  _th_llm="ANTHROPIC_API_KEY"
fi
_th_obs="unset"
if [[ -n "${OBS_PASSWORD:-}" ]]; then
  _th_obs="set"
fi

echo "env.sh: LLM credential=${_th_llm}; OBS_PASSWORD=${_th_obs}; ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL:-<default>}"
unset _th_root _th_llm _th_obs
