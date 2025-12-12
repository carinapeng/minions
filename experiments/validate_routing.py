"""
Routing Validation Evaluation

This script evaluates whether routing decisions are CORRECT by:
1. For each query, router predicts local or remote
2. Run BOTH local and remote models
3. Check which model(s) give correct answers
4. Validate: Did router choose a model that gives correct answer?

Key metrics:
- Routing Precision: When router says "local", how often does local give correct answer?
- Routing Recall: Of queries local CAN handle, how many did router route locally?
- Overall Accuracy: % of queries where routed model gives correct answer

Usage:
    python experiments/validate_routing.py --queries test_data.py --max-queries 100
"""

import argparse
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.test_data import TestDataset, TestQuery
from minions.utils.complexity_scorer import ComplexityScorer
from minions.clients import OllamaClient, TogetherClient


@dataclass
class ValidationResult:
    """Result for validating one routing decision"""
    query: str
    query_type: str
    expected_route: str  # From manual labeling

    # Complexity analysis
    complexity_score: float
    erotetic_type: str
    predicted_route: str  # From router
    router_confidence: float

    # Ground truth execution
    local_answer: str
    local_time: float
    local_correct: bool

    remote_answer: str
    remote_time: float
    remote_correct: bool

    # Validation metrics
    routing_correct: bool  # Did router pick a model that gives correct answer?
    routing_optimal: bool  # Did router pick the BEST option (fastest correct model)?
    quality_preserved: bool  # Would routed answer be correct?

    ground_truth: str
    judge_reasoning: str


class RoutingValidator:
    """Validate routing decisions by running both models"""

    def __init__(self, local_client, remote_client, judge_client):
        self.local_client = local_client
        self.remote_client = remote_client
        self.judge_client = judge_client

        # Router components
        self.complexity_scorer = ComplexityScorer()

        # Separate client for self-consistency
        self.sc_client = OllamaClient(model_name="llama3.2", temperature=0.7, max_tokens=200, use_async=False)

        # Adaptive thresholds
        self.thresholds = {
            'factual': 0.7,
            'conceptual': 0.6,
            'procedural': 0.55,
            'causal': 0.5,
            'comparative': 0.45,
            'evaluative': 0.4,
            'synthetic': 0.3,
            'metacognitive': 0.4
        }

    def predict_route(self, query: str, context: Optional[List[str]] = None) -> Tuple[str, float, float]:
        """Predict routing decision"""
        # Complexity analysis
        complexity_score_obj = self.complexity_scorer.score(query, context)
        complexity_score = complexity_score_obj.overall

        # Self-consistency (simplified for speed - can enable for full eval)
        sc_uncertainty = 0.5  # Default

        # Combined score
        combined = 0.6 * complexity_score + 0.4 * sc_uncertainty

        threshold = self.thresholds.get(complexity_score_obj.erotetic_type.value, 0.5)

        # Safety override
        if sc_uncertainty > 0.8:
            route = "remote"
            confidence = sc_uncertainty
        elif combined < threshold:
            route = "local"
            confidence = (threshold - combined) / threshold
        else:
            route = "remote"
            confidence = (combined - threshold) / (1.0 - threshold)

        return route, confidence, complexity_score

    def run_local(self, query: str, context: Optional[List[str]] = None) -> Tuple[str, float]:
        """Run query on local model"""
        start = time.time()
        try:
            messages = [{"role": "user", "content": query}]
            if context:
                context_str = "\n\n".join(context)
                messages[0]["content"] = f"Context:\n{context_str}\n\nQuery: {query}"

            result = self.local_client.chat(messages=messages)
            if isinstance(result, tuple):
                answer = result[0][0] if isinstance(result[0], list) else result[0]
            else:
                answer = result[0] if isinstance(result, list) else result

            elapsed = time.time() - start
            return answer.strip(), elapsed
        except Exception as e:
            return f"Error: {e}", time.time() - start

    def run_remote(self, query: str, context: Optional[List[str]] = None) -> Tuple[str, float]:
        """Run query on remote model"""
        start = time.time()
        try:
            messages = [{"role": "user", "content": query}]
            if context:
                context_str = "\n\n".join(context)
                messages[0]["content"] = f"Context:\n{context_str}\n\nQuery: {query}"

            result = self.remote_client.chat(messages=messages)
            if isinstance(result, tuple):
                answer = result[0][0] if isinstance(result[0], list) else result[0]
            else:
                answer = result[0] if isinstance(result, list) else result

            elapsed = time.time() - start
            return answer.strip(), elapsed
        except Exception as e:
            return f"Error: {e}", time.time() - start

    def judge_correctness(self, query: str, answer: str, ground_truth: str) -> Tuple[bool, str]:
        """Judge if answer is correct compared to ground truth"""
        prompt = f"""Evaluate if the answer is CORRECT compared to ground truth.

Question: {query}

Ground Truth: {ground_truth}

Answer to Evaluate: {answer}

Instructions:
- Check if the answer is factually correct and matches the ground truth
- Minor wording differences are OK if the meaning is the same
- If answer has errors or is significantly wrong, mark as incorrect

Response Format (JSON only):
{{"correct": true/false, "reasoning": "<brief explanation>"}}

Your response:"""

        try:
            result = self.judge_client.chat([{"role": "user", "content": prompt}])
            if isinstance(result, tuple):
                response = result[0][0] if isinstance(result[0], list) else result[0]
            else:
                response = result[0] if isinstance(result, list) else result

            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
                correct = data.get("correct", False)
                reasoning = data.get("reasoning", "")
                return bool(correct), reasoning
        except Exception as e:
            print(f"Judge error: {e}")

        return False, "Could not judge"

    def validate(self, test_query: TestQuery) -> ValidationResult:
        """Validate routing decision for one query"""
        print(f"\n{'='*80}")
        print(f"Query: {test_query.query}")
        print(f"Type: {test_query.query_type}")
        print(f"{'='*80}")

        # 1. Router prediction
        print("\n[1/4] ROUTER PREDICTION...")
        predicted_route, confidence, complexity = self.predict_route(test_query.query, test_query.context)
        complexity_obj = self.complexity_scorer.score(test_query.query, test_query.context)
        erotetic_type = complexity_obj.erotetic_type.value

        print(f"  Predicted: {predicted_route.upper()}")
        print(f"  Complexity: {complexity:.3f} ({erotetic_type})")
        print(f"  Confidence: {confidence:.2%}")

        # 2. Run LOCAL model
        print("\n[2/4] RUNNING LOCAL MODEL...")
        local_answer, local_time = self.run_local(test_query.query, test_query.context)
        print(f"  Time: {local_time:.2f}s")
        print(f"  Answer: {local_answer[:100]}...")

        # Judge local answer
        local_correct, local_reasoning = self.judge_correctness(
            test_query.query, local_answer, test_query.ground_truth
        )
        print(f"  Correct: {'✓' if local_correct else '✗'} - {local_reasoning}")

        # 3. Run REMOTE model
        print("\n[3/4] RUNNING REMOTE MODEL...")
        remote_answer, remote_time = self.run_remote(test_query.query, test_query.context)
        print(f"  Time: {remote_time:.2f}s")
        print(f"  Answer: {remote_answer[:100]}...")

        # Judge remote answer
        remote_correct, remote_reasoning = self.judge_correctness(
            test_query.query, remote_answer, test_query.ground_truth
        )
        print(f"  Correct: {'✓' if remote_correct else '✗'} - {remote_reasoning}")

        # 4. Validate routing decision
        print("\n[4/4] VALIDATING ROUTING...")

        # Did router pick a model that gives correct answer?
        if predicted_route == "local":
            quality_preserved = local_correct
            judge_reasoning = local_reasoning
        else:
            quality_preserved = remote_correct
            judge_reasoning = remote_reasoning

        # Is the routing decision correct? (picked a model that answers correctly)
        routing_correct = quality_preserved

        # Is routing optimal? (picked the fastest model among correct ones)
        if local_correct and remote_correct:
            # Both correct - optimal is local (faster)
            routing_optimal = (predicted_route == "local")
        elif local_correct and not remote_correct:
            # Only local correct - must pick local
            routing_optimal = (predicted_route == "local")
        elif not local_correct and remote_correct:
            # Only remote correct - must pick remote
            routing_optimal = (predicted_route == "remote")
        else:
            # Both wrong - can't win
            routing_optimal = False

        speedup = remote_time / local_time if local_time > 0 else 0

        print(f"  Routing Correct: {'✓' if routing_correct else '✗'}")
        print(f"  Routing Optimal: {'✓' if routing_optimal else '✗'}")
        print(f"  Quality Preserved: {'✓' if quality_preserved else '✗'}")
        if routing_optimal and predicted_route == "local":
            print(f"  Speedup Achieved: {speedup:.1f}x")

        return ValidationResult(
            query=test_query.query,
            query_type=test_query.query_type,
            expected_route=test_query.expected_route,
            complexity_score=complexity,
            erotetic_type=erotetic_type,
            predicted_route=predicted_route,
            router_confidence=confidence,
            local_answer=local_answer,
            local_time=local_time,
            local_correct=local_correct,
            remote_answer=remote_answer,
            remote_time=remote_time,
            remote_correct=remote_correct,
            routing_correct=routing_correct,
            routing_optimal=routing_optimal,
            quality_preserved=quality_preserved,
            ground_truth=test_query.ground_truth,
            judge_reasoning=judge_reasoning
        )


def analyze_results(results: List[ValidationResult]) -> Dict[str, Any]:
    """Comprehensive analysis of routing validation"""

    print(f"\n{'='*80}")
    print("ROUTING VALIDATION ANALYSIS")
    print(f"{'='*80}")

    total = len(results)

    # Overall metrics
    routing_correct = sum(1 for r in results if r.routing_correct)
    routing_optimal = sum(1 for r in results if r.routing_optimal)
    quality_preserved = sum(1 for r in results if r.quality_preserved)

    print(f"\n📊 OVERALL METRICS")
    print(f"Total Queries: {total}")
    print(f"Routing Correct: {routing_correct}/{total} ({routing_correct/total*100:.1f}%)")
    print(f"  (Router picked a model that gives correct answer)")
    print(f"Routing Optimal: {routing_optimal}/{total} ({routing_optimal/total*100:.1f}%)")
    print(f"  (Router picked the BEST model - fastest among correct ones)")
    print(f"Quality Preserved: {quality_preserved}/{total} ({quality_preserved/total*100:.1f}%)")
    print(f"  (Routed answer is correct)")

    # Local vs Remote performance
    local_total = sum(1 for r in results if r.predicted_route == "local")
    remote_total = sum(1 for r in results if r.predicted_route == "remote")

    print(f"\n📊 ROUTING DISTRIBUTION")
    print(f"Routed to Local: {local_total}/{total} ({local_total/total*100:.1f}%)")
    print(f"Routed to Remote: {remote_total}/{total} ({remote_total/total*100:.1f}%)")

    # Precision: When router says "local", how often is local correct?
    local_results = [r for r in results if r.predicted_route == "local"]
    if local_results:
        local_precision = sum(1 for r in local_results if r.local_correct) / len(local_results)
        print(f"\n📊 LOCAL ROUTING PRECISION")
        print(f"When routed to LOCAL, local model correct: {sum(1 for r in local_results if r.local_correct)}/{len(local_results)} ({local_precision*100:.1f}%)")

    # Recall: Of queries local CAN handle, how many routed to local?
    can_handle_local = [r for r in results if r.local_correct]
    if can_handle_local:
        local_recall = sum(1 for r in can_handle_local if r.predicted_route == "local") / len(can_handle_local)
        print(f"\n📊 LOCAL ROUTING RECALL")
        print(f"Of queries local CAN handle, routed to local: {sum(1 for r in can_handle_local if r.predicted_route == 'local')}/{len(can_handle_local)} ({local_recall*100:.1f}%)")

    # Efficiency analysis
    optimal_local = [r for r in results if r.routing_optimal and r.predicted_route == "local"]
    if optimal_local:
        avg_speedup = np.mean([r.remote_time / r.local_time for r in optimal_local])
        total_time_saved = sum(r.remote_time - r.local_time for r in optimal_local)
        print(f"\n⚡ EFFICIENCY GAINS")
        print(f"Optimal local routes: {len(optimal_local)}")
        print(f"Average speedup: {avg_speedup:.1f}x")
        print(f"Total time saved: {total_time_saved:.1f}s")

    # Error analysis
    errors = [r for r in results if not r.routing_correct]
    if errors:
        print(f"\n❌ ROUTING ERRORS ({len(errors)} queries)")
        false_local = [r for r in errors if r.predicted_route == "local"]
        false_remote = [r for r in errors if r.predicted_route == "remote"]

        if false_local:
            print(f"\n  False LOCAL ({len(false_local)} queries):")
            print(f"    Router said LOCAL but local gave wrong answer")
            for r in false_local[:3]:
                print(f"      - {r.query[:60]}...")
                print(f"        Complexity: {r.complexity_score:.3f}, Type: {r.erotetic_type}")

        if false_remote:
            print(f"\n  False REMOTE ({len(false_remote)} queries):")
            print(f"    Router said REMOTE but remote gave wrong answer")
            for r in false_remote[:3]:
                print(f"      - {r.query[:60]}...")

    # By query type
    by_type = defaultdict(lambda: {"total": 0, "correct": 0, "optimal": 0})
    for r in results:
        by_type[r.query_type]["total"] += 1
        if r.routing_correct:
            by_type[r.query_type]["correct"] += 1
        if r.routing_optimal:
            by_type[r.query_type]["optimal"] += 1

    print(f"\n📊 PERFORMANCE BY QUERY TYPE")
    for qtype in sorted(by_type.keys()):
        stats = by_type[qtype]
        acc = stats["correct"] / stats["total"] * 100
        opt = stats["optimal"] / stats["total"] * 100
        print(f"  {qtype:12s}: {stats['correct']}/{stats['total']} correct ({acc:.1f}%), {stats['optimal']}/{stats['total']} optimal ({opt:.1f}%)")

    return {
        "total_queries": total,
        "routing_correct": routing_correct,
        "routing_optimal": routing_optimal,
        "quality_preserved": quality_preserved,
        "routing_accuracy": routing_correct / total,
        "optimality": routing_optimal / total,
        "local_precision": local_precision if local_results else 0,
        "local_recall": local_recall if can_handle_local else 0,
        "avg_speedup": float(avg_speedup) if optimal_local else 0,
        "by_type": {k: v for k, v in by_type.items()}
    }


def create_visualizations(results: List[ValidationResult], analysis: Dict[str, Any]):
    """Create comprehensive visualizations"""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Overall metrics
    ax1 = fig.add_subplot(gs[0, :])
    metrics = ['Routing\nCorrect', 'Routing\nOptimal', 'Quality\nPreserved']
    values = [
        analysis['routing_accuracy'] * 100,
        analysis['optimality'] * 100,
        analysis['quality_preserved'] / analysis['total_queries'] * 100
    ]
    bars = ax1.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black', linewidth=2)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('Overall Routing Validation Metrics', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 100)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 2. Routing distribution
    ax2 = fig.add_subplot(gs[1, 0])
    local_count = sum(1 for r in results if r.predicted_route == "local")
    remote_count = sum(1 for r in results if r.predicted_route == "remote")
    ax2.pie([local_count, remote_count], labels=['Local', 'Remote'],
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
    ax2.set_title('Routing Distribution', fontsize=12, fontweight='bold')

    # 3. Local precision vs recall
    ax3 = fig.add_subplot(gs[1, 1])
    precision_recall = ['Precision\n(When say local,\nlocal correct)', 'Recall\n(Of can-local,\nroute local)']
    pr_values = [analysis['local_precision'] * 100, analysis['local_recall'] * 100]
    bars = ax3.bar(precision_recall, pr_values, color=['#9b59b6', '#f39c12'], edgecolor='black', linewidth=2)
    ax3.set_ylabel('Percentage (%)', fontsize=11)
    ax3.set_title('Local Routing Precision & Recall', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, 100)

    for bar, val in zip(bars, pr_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. Speedup distribution
    ax4 = fig.add_subplot(gs[1, 2])
    optimal_local = [r for r in results if r.routing_optimal and r.predicted_route == "local"]
    if optimal_local:
        speedups = [r.remote_time / r.local_time for r in optimal_local]
        ax4.hist(speedups, bins=15, color='#2ecc71', edgecolor='black', alpha=0.7)
        ax4.axvline(np.mean(speedups), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(speedups):.1f}x')
        ax4.set_xlabel('Speedup Factor', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Speedup Distribution (Optimal Local Routes)', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

    # 5. Performance by query type
    ax5 = fig.add_subplot(gs[2, :])
    types = sorted(analysis['by_type'].keys())
    correct_pct = [analysis['by_type'][t]['correct'] / analysis['by_type'][t]['total'] * 100 for t in types]
    optimal_pct = [analysis['by_type'][t]['optimal'] / analysis['by_type'][t]['total'] * 100 for t in types]

    x = np.arange(len(types))
    width = 0.35

    bars1 = ax5.bar(x - width/2, correct_pct, width, label='Routing Correct', color='#3498db', edgecolor='black')
    bars2 = ax5.bar(x + width/2, optimal_pct, width, label='Routing Optimal', color='#2ecc71', edgecolor='black')

    ax5.set_xlabel('Query Type', fontsize=12)
    ax5.set_ylabel('Percentage (%)', fontsize=12)
    ax5.set_title('Routing Performance by Query Type', fontsize=13, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(types, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    ax5.set_ylim(0, 100)

    plt.suptitle('Routing Validation: Complete Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('routing_validation.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: routing_validation.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Validate routing decisions")
    parser.add_argument("--max-queries", type=int, default=100, help="Max queries to validate")
    parser.add_argument("--output", type=str, default="routing_validation.json", help="Output file")
    parser.add_argument("--query-types", type=str, nargs="+", default=None, help="Specific query types")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N queries")
    parser.add_argument("--queries-file", type=str, default=None,
                       help="Load queries from JSON file (merged_queries.json or generated_queries.json)")

    args = parser.parse_args()

    print("="*80)
    print("ROUTING VALIDATION EVALUATION")
    print("="*80)
    print("\nThis validates routing decisions by running BOTH models")
    print("and checking if router chose correctly.\n")

    # Initialize clients
    print("Initializing clients...")
    local_client = OllamaClient(model_name="llama3.2", temperature=0.0, max_tokens=4096, use_async=False)
    remote_client = TogetherClient(model="Qwen/Qwen2.5-72B-Instruct-Turbo")
    judge_client = TogetherClient(model="Qwen/Qwen2.5-72B-Instruct-Turbo")
    print("✓ Clients ready\n")

    # Load queries
    if args.queries_file:
        print(f"Loading queries from {args.queries_file}...")
        with open(args.queries_file, 'r') as f:
            data = json.load(f)

        query_dicts = data.get("queries", [])

        # Convert to TestQuery objects
        queries = []
        for q in query_dicts:
            queries.append(TestQuery(
                query=q["query"],
                query_type=q["query_type"],
                expected_route=q["expected_route"],
                ground_truth=q["ground_truth"],
                context=q.get("context", []),
                difficulty=q.get("difficulty", "medium")
            ))

        print(f"✓ Loaded {len(queries)} queries from JSON")
    else:
        print("Loading queries from test_data.py...")
        dataset = TestDataset()
        if args.query_types:
            queries = []
            for qt in args.query_types:
                queries.extend(dataset.get_by_type(qt))
        else:
            queries = dataset.get_all()

        print(f"✓ Loaded {len(queries)} queries from test_data.py")

    # Skip first N queries if requested
    if args.skip > 0:
        print(f"\n⏭️  Skipping first {args.skip} queries...")
        queries = queries[args.skip:]
        print(f"✓ {len(queries)} queries remaining after skip")

    # Limit to max_queries
    if len(queries) > args.max_queries:
        queries = queries[:args.max_queries]
        print(f"✓ Limited to {args.max_queries} queries\n")
    else:
        print(f"✓ Processing all {len(queries)} queries\n")

    # Validate
    validator = RoutingValidator(local_client, remote_client, judge_client)

    results = []
    for i, query in enumerate(queries, 1):
        print(f"\n{'#'*80}")
        print(f"# Query {i}/{len(queries)}")
        print(f"{'#'*80}")

        result = validator.validate(query)
        results.append(result)

    # Analyze
    analysis = analyze_results(results)

    # Visualize
    create_visualizations(results, analysis)

    # Save
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "analysis": analysis,
        "results": [asdict(r) for r in results]
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {args.output}")
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
