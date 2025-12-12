"""
Merge generated queries with existing test dataset

This script combines:
1. Existing 37 manually curated queries (test_data.py)
2. Newly generated queries (generated_queries.json)

Into a unified dataset for evaluation.

Usage:
    python experiments/merge_queries.py --generated generated_queries.json --output merged_queries.json
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.test_data import TestDataset


def main():
    parser = argparse.ArgumentParser(description="Merge query datasets")
    parser.add_argument("--generated", type=str, default="generated_queries.json",
                       help="Generated queries JSON file")
    parser.add_argument("--output", type=str, default="merged_queries.json",
                       help="Output merged dataset")

    args = parser.parse_args()

    print("="*80)
    print("MERGING QUERY DATASETS")
    print("="*80)

    # Load existing queries
    print("\nLoading existing test dataset...")
    dataset = TestDataset()
    existing_queries = dataset.get_all()
    print(f"✓ Loaded {len(existing_queries)} existing queries")

    # Load generated queries
    print(f"\nLoading generated queries from {args.generated}...")
    with open(args.generated, 'r') as f:
        generated_data = json.load(f)

    generated_queries = generated_data.get("queries", [])
    print(f"✓ Loaded {len(generated_queries)} generated queries")

    # Convert existing queries to dict format
    existing_dicts = []
    for q in existing_queries:
        existing_dicts.append({
            "query": q.query,
            "query_type": q.query_type,
            "expected_route": q.expected_route,
            "ground_truth": q.ground_truth,
            "context": q.context,
            "difficulty": q.difficulty
        })

    # Merge
    all_queries = existing_dicts + generated_queries

    # Statistics
    by_type = {}
    by_route = {}
    by_difficulty = {}

    for q in all_queries:
        qtype = q["query_type"]
        route = q["expected_route"]
        diff = q.get("difficulty", "medium")

        by_type[qtype] = by_type.get(qtype, 0) + 1
        by_route[route] = by_route.get(route, 0) + 1
        by_difficulty[diff] = by_difficulty.get(diff, 0) + 1

    # Save merged dataset
    output_data = {
        "metadata": {
            "total_queries": len(all_queries),
            "existing_queries": len(existing_dicts),
            "generated_queries": len(generated_queries),
            "distribution": {
                "by_type": by_type,
                "by_route": by_route,
                "by_difficulty": by_difficulty
            }
        },
        "queries": all_queries
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print("MERGE COMPLETE")
    print(f"{'='*80}")
    print(f"\nTotal queries: {len(all_queries)}")
    print(f"  Existing: {len(existing_dicts)}")
    print(f"  Generated: {len(generated_queries)}")
    print(f"\nBy type: {by_type}")
    print(f"By route: {by_route}")
    print(f"By difficulty: {by_difficulty}")
    print(f"\n✓ Saved to {args.output}")


if __name__ == "__main__":
    main()
