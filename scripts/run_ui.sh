#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../musicbrain/ui"
exec npm run dev
