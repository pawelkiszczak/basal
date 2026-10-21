"""Check JSONL syntax without importing basal or changing files."""
import json
import sys


def main(filenames):
    failed = False
    for filename in filenames:
        try:
            with open(filename, encoding="utf-8") as source:
                for number, line in enumerate(source, 1):
                    if not line.strip():  # The batch runner also skips blank lines.
                        continue
                    try:
                        json.loads(line)
                    except ValueError as error:
                        print(f"{filename}:{number}: {error}")
                        failed = True
        except (OSError, UnicodeError) as error:
            print(f"{filename}: {error}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
