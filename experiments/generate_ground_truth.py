"""
Generate Empirical Ground Truth for Routing Decisions

This script runs all queries through BOTH local and full protocol,
then uses LLM judge to determine which route should be preferred.

This creates an empirical ground truth dataset for evaluating the router.

Usage:
    python experiments/generate_ground_truth.py --max-queries 37 --output ground_truth.json
"""

import argparse
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.test_data import TestDataset, TestQuery
from minions.minions import Minions


@dataclass
class GroundTruthLabel:
    """Empirically-derived ground truth for routing"""
    query: str
    query_type: str

    # Empirical results
    local_answer: str
    local_time: float
    full_answer: str
    full_time: float

    # Judge evaluation
    judge_choice: Optional[int]  # 0=local, 1=full, None=tie
    judge_reason: str
    quality_gap: str  # "none", "minor", "significant"

    # Ground truth decision
    ground_truth_route: str  # "local" or "remote"
    rationale: str

    # Metrics
    speedup_if_local: float
    quality_preserved: bool


class GroundTruthGenerator:
    """Generate empirical ground truth by running head-to-head comparisons"""

    def __init__(self, local_client, remote_client, judge_client):
        self.local_client = local_client
        self.remote_client = remote_client
        self.judge_client = judge_client

    def run_local(self, test_query: TestQuery) -> tuple[str, float]:
        """Run query with local-only"""
        start = time.time()
        try:
            messages = [{"role": "user", "content": test_query.query}]
            if test_query.context:
                context_str = "\n\n".join(test_query.context)
                messages[0]["content"] = f"Context:\n{context_str}\n\nQuery: {test_query.query}"

            result = self.local_client.chat(messages=messages)
            if isinstance(result, tuple):
                answer = result[0][0] if isinstance(result[0], list) else result[0]
            else:
                answer = result[0] if isinstance(result, list) else result

            elapsed = time.time() - start
            return answer.strip(), elapsed
        except Exception as e:
            return f"Error: {e}", time.time() - start

    def run_full_protocol(self, test_query: TestQuery) -> tuple[str, float]:
        """Run query with full Minions protocol"""
        start = time.time()
        try:
            minions = Minions(
                local_client=self.local_client,
                remote_client=self.remote_client,
                mode="smart_routing"
            )
            result = minions(
                task=test_query.query,
                doc_metadata="query",
                context=test_query.context if test_query.context else [],
                max_rounds=3
            )
            answer = result.get("final_answer", "") if isinstance(result, dict) else str(result)
            elapsed = time.time() - start
            return answer.strip(), elapsed
        except Exception as e:
            return f"Error: {e}", time.time() - start

    def judge(self, query: str, local_ans: str, full_ans: str, ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """LLM judge comparison"""
        gt_context = f"\n\nGround Truth Reference:\n{ground_truth}\n" if ground_truth else ""

        prompt = f"""You are an expert evaluator comparing two AI-generated answers.

Question:
{query}
{gt_context}

Candidate 0 (Local Model):
{local_ans}

Candidate 1 (Full Protocol):
{full_ans}

Instructions:
1. Evaluate QUALITY: correctness, completeness, accuracy
2. Determine QUALITY GAP:
   - "none": Both answers are equally good OR both wrong
   - "minor": One is slightly better, but difference is small
   - "significant": One is clearly superior

Response Format (JSON only):
{{"choice": <0, 1, or null>, "reason": "<one sentence>", "quality_gap": "<none/minor/significant>"}}

Your response:"""

        try:
            result = self.judge_client.chat([{"role": "user", "content": prompt}])
            if len(result) == 3:
                response, usage, _ = result
            else:
                response, usage = result

            raw = response[0]
            import re
            json_match = re.search(r'\{[^}]+\}', raw)
            if json_match:
                data = json.loads(json_match.group())
                choice = data.get("choice")
                if choice == "null" or choice == "None":
                    choice = None
                elif choice is not None:
                    choice = int(choice)
                reason = data.get("reason", "")
                quality_gap = data.get("quality_gap", "unknown")
                return {"choice": choice, "reason": reason, "quality_gap": quality_gap, "raw": raw}
        except Exception as e:
            print(f"Judge error: {e}")

        return {"choice": None, "reason": f"Error", "quality_gap": "unknown", "raw": ""}

    def generate_ground_truth(self, test_query: TestQuery) -> GroundTruthLabel:
        """Generate empirical ground truth for one query"""
        print(f"\n{'='*80}")
        print(f"Query: {test_query.query}")
        print(f"Type: {test_query.query_type}")
        print(f"{'='*80}")

        # Run both routes
        print("\n[1/3] Running LOCAL...")
        local_answer, local_time = self.run_local(test_query)
        print(f"  ✓ Completed in {local_time:.2f}s")
        print(f"  Answer: {local_answer[:150]}...")

        print("\n[2/3] Running FULL PROTOCOL...")
        full_answer, full_time = self.run_full_protocol(test_query)
        print(f"  ✓ Completed in {full_time:.2f}s")
        print(f"  Answer: {full_answer[:150]}...")

        # Judge comparison
        print("\n[3/3] LLM JUDGE EVALUATION...")
        judge_result = self.judge(test_query.query, local_answer, full_answer, test_query.ground_truth)
        choice = judge_result["choice"]
        reason = judge_result["reason"]
        quality_gap = judge_result["quality_gap"]

        print(f"  Choice: {choice} (0=local, 1=full, None=tie)")
        print(f"  Reason: {reason}")
        print(f"  Quality Gap: {quality_gap}")

        # Determine ground truth routing decision
        speedup = full_time / local_time if local_time > 0 else 0

        # Decision logic
        if choice == 0:
            # Judge prefers local - clear win for local
            gt_route = "local"
            rationale = f"Local preferred by judge with {speedup:.1f}x speedup"
            quality_preserved = True
        elif choice is None:
            # Tie - prefer local for efficiency
            gt_route = "local"
            rationale = f"Quality tie - prefer local for {speedup:.1f}x speedup"
            quality_preserved = True
        else:
            # Judge prefers full
            if quality_gap == "minor" and speedup > 10:
                # Accept minor quality loss for huge speedup
                gt_route = "local"
                rationale = f"Minor quality gap acceptable for {speedup:.1f}x speedup"
                quality_preserved = False
            elif quality_gap == "significant":
                # Significant quality gap - must use full
                gt_route = "remote"
                rationale = f"Significant quality gap requires full protocol"
                quality_preserved = False
            else:
                # Unknown quality gap or moderate - be conservative
                gt_route = "remote"
                rationale = f"Quality advantage to full protocol"
                quality_preserved = False

        print(f"\n  → GROUND TRUTH: {gt_route.upper()}")
        print(f"  → Rationale: {rationale}")

        return GroundTruthLabel(
            query=test_query.query,
            query_type=test_query.query_type,
            local_answer=local_answer,
            local_time=local_time,
            full_answer=full_answer,
            full_time=full_time,
            judge_choice=choice,
            judge_reason=reason,
            quality_gap=quality_gap,
            ground_truth_route=gt_route,
            rationale=rationale,
            speedup_if_local=speedup,
            quality_preserved=quality_preserved
        )


def main():
    parser = argparse.ArgumentParser(description="Generate empirical ground truth for routing")
    parser.add_argument("--max-queries", type=int, default=37, help="Max queries to process")
    parser.add_argument("--output", type=str, default="ground_truth.json", help="Output file")
    parser.add_argument("--query-types", type=str, nargs="+", default=None, help="Specific query types")

    args = parser.parse_args()

    print("="*80)
    print("EMPIRICAL GROUND TRUTH GENERATION")
    print("="*80)
    print("\nThis script runs head-to-head comparison: LOCAL vs FULL PROTOCOL")
    print("Judge evaluation determines which route should be preferred")
    print()

    # Initialize clients
    print("Initializing clients...")
    from minions.clients import OllamaClient, TogetherClient

    local_client = OllamaClient(model_name="llama3.2", temperature=0.0, max_tokens=4096, use_async=False)
    remote_client = TogetherClient(model="Qwen/Qwen2.5-72B-Instruct-Turbo")
    judge_client = TogetherClient(model="Qwen/Qwen2.5-72B-Instruct-Turbo")

    print("✓ Clients ready\n")

    # Load test data
    dataset = TestDataset()
    if args.query_types:
        queries = []
        for qt in args.query_types:
            queries.extend(dataset.get_by_type(qt))
    else:
        queries = dataset.get_all()

    queries = queries[:args.max_queries]
    print(f"✓ Loaded {len(queries)} queries\n")

    # Generate ground truth
    generator = GroundTruthGenerator(local_client, remote_client, judge_client)

    results = []
    for i, query in enumerate(queries, 1):
        print(f"\n{'#'*80}")
        print(f"# Query {i}/{len(queries)}")
        print(f"{'#'*80}")

        gt_label = generator.generate_ground_truth(query)
        results.append(gt_label)

    # Analysis
    print(f"\n{'='*80}")
    print("GROUND TRUTH SUMMARY")
    print(f"{'='*80}")

    local_routes = sum(1 for r in results if r.ground_truth_route == "local")
    remote_routes = sum(1 for r in results if r.ground_truth_route == "remote")

    print(f"\nTotal queries: {len(results)}")
    print(f"Ground truth LOCAL: {local_routes} ({local_routes/len(results)*100:.1f}%)")
    print(f"Ground truth REMOTE: {remote_routes} ({remote_routes/len(results)*100:.1f}%)")

    # Breakdown by query type
    from collections import defaultdict
    by_type = defaultdict(lambda: {"local": 0, "remote": 0})
    for r in results:
        by_type[r.query_type][r.ground_truth_route] += 1

    print("\nBreakdown by query type:")
    for qtype in sorted(by_type.keys()):
        local_ct = by_type[qtype]["local"]
        remote_ct = by_type[qtype]["remote"]
        total = local_ct + remote_ct
        print(f"  {qtype:12s}: {local_ct}/{total} local, {remote_ct}/{total} remote")

    # Quality preservation
    quality_preserved = sum(1 for r in results if r.quality_preserved)
    print(f"\nQuality preserved: {quality_preserved}/{len(results)} ({quality_preserved/len(results)*100:.1f}%)")

    avg_speedup = sum(r.speedup_if_local for r in results) / len(results)
    print(f"Average speedup (if local): {avg_speedup:.1f}x")

    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_queries": len(results),
            "local_routes": local_routes,
            "remote_routes": remote_routes,
            "quality_preserved": quality_preserved,
            "avg_speedup": float(avg_speedup)
        },
        "ground_truth_labels": [asdict(r) for r in results]
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Ground truth saved to {args.output}")
    print(f"\nYou can now evaluate your router against this empirical ground truth!")


if __name__ == "__main__":
    main()
