#!/usr/bin/env python3
"""Capture a foreground command's output in bounded, rotating log files."""

import argparse
import codecs
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import signal
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=5 * 1024 * 1024)
    parser.add_argument("--backups", type=int, default=2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.max_bytes < 65536 or not 1 <= args.backups <= 10:
        parser.error("Provide a command, max-bytes >= 65536 and 1-10 backups")
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        args.log_file,
        maxBytes=args.max_bytes,
        backupCount=args.backups,
        encoding="utf-8",
        errors="replace",
    )
    handler.terminator = ""
    try:
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
        ) as child:

            def forward(signum: int, _frame: object) -> None:
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    pass

            signal.signal(signal.SIGTERM, forward)
            signal.signal(signal.SIGINT, forward)
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            assert child.stdout is not None
            while chunk := child.stdout.read1(8192):
                handler.emit(logging.LogRecord("capture", logging.INFO, "", 0, decoder.decode(chunk), (), None))
            tail = decoder.decode(b"", final=True)
            if tail:
                handler.emit(logging.LogRecord("capture", logging.INFO, "", 0, tail, (), None))
            returncode = child.wait()
            return returncode if returncode >= 0 else 128 - returncode
    finally:
        handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
