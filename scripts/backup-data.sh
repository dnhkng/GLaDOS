#!/bin/sh
# Dated archive of GLaDOS's state: data/ (settings, knowledge, console pairing, certificates) and
# ~/.glados (long-term memory). Logs are left out. Keeps the newest GLADOS_BACKUP_KEEP archives.
set -eu
umask 077  # archives hold the pairing token and the local certificate authority key
repo="$(cd "$(dirname "$0")/.." && pwd)"
dest="${GLADOS_BACKUP_DIR:-$HOME/.local/share/glados-backups}"
keep="${GLADOS_BACKUP_KEEP:-14}"
mkdir -p "$dest"
archive="$dest/glados-$(date +%Y-%m-%d_%H%M%S).tar.gz"
set -- -C "$repo" --exclude=data/logs data
[ -d "$HOME/.glados" ] && set -- "$@" -C "$HOME" .glados
tar -czf "$archive" "$@"
ls -1t "$dest"/glados-*.tar.gz | tail -n +"$((keep + 1))" | xargs -r rm --
echo "$archive"
