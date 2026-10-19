"""JSON log ingestion for batch jobs and shell pipelines."""

import argparse
import json
from pathlib import Path
import sys

from epiplexity.estimators import PrequentialEstimator, RequentialEstimator


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Estimate code lengths from training logs")
    parser.add_argument("method", choices=("prequential", "requential"))
    parser.add_argument("input", help="JSON file, or - for stdin")
    args = parser.parse_args(argv)
    try:
        text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("input must be a JSON object")
        metadata = payload.pop("metadata", None)
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        estimator = PrequentialEstimator if args.method == "prequential" else RequentialEstimator
        output = estimator.estimate(**payload).to_json(metadata=metadata)
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        print(f"epiplexity: error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0
