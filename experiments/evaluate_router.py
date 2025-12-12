"""
Evaluate Smart Router Against Empirical Ground Truth

This script evaluates the complexity-based router's decisions against
empirical ground truth generated from head-to-head comparisons.

Usage:
    # First generate ground truth:
    python experiments/generate_ground_truth.py --max-queries 37

    # Then evaluate router:
    python experiments/evaluate_router.py --ground-truth ground_truth.json
"""

import argparse
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.utils.complexity_scorer import ComplexityScorer


@dataclass
class RouterEvaluation:
    """Evaluation of router decision vs ground truth"""
    query: str
    query_type: str

    # Ground truth
    ground_truth_route: str
    ground_truth_rationale: str

    # Router decision
    complexity_score: float
    erotetic_type: str
    bloom_level: str
    reasoning_depth: int
    self_consistency_uncertainty: float
    predicted_route: str
    router_confidence: float

    # Evaluation
    correct: bool
    speedup_if_correct: float
    quality_preserved: bool


class RouterEvaluator:
    """Evaluate routing decisions against ground truth"""

    def __init__(self, local_client, self_consistency_k: int = 3):
        self.complexity_scorer = ComplexityScorer()
        self.self_consistency_k = self_consistency_k

        # Separate client for self-consistency
        from minions.clients import OllamaClient
        self.sc_client = OllamaClient(model_name="llama3.2", temperature=0.7, max_tokens=200, use_async=False)

        # Adaptive thresholds (same as enhanced router)
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

    def _measure_self_consistency(self, query: str, context: Optional[List[str]] = None) -> float:
        """Measure uncertainty via self-consistency"""
        if self.self_consistency_k < 2:
            return 0.0

        try:
            responses = []
            messages = [{"role": "user", "content": query}]
            if context:
                context_str = "\n\n".join(context)
                messages[0]["content"] = f"Context:\n{context_str}\n\nQuery: {query}"

            for _ in range(self.self_consistency_k):
                result = self.sc_client.chat(messages=messages)
                if isinstance(result, tuple):
                    response = result[0][0] if isinstance(result[0], list) else result[0]
                else:
                    response = result[0] if isinstance(result, list) else result
                responses.append(response.strip().lower())

            unique = len(set(responses))
            uncertainty = unique / len(responses)
            return uncertainty
        except:
            return 0.5

    def predict_route(self, query: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """Predict routing decision using complexity scorer"""
        start_time = time.time()

        # 1. Complexity scoring
        complexity_score = self.complexity_scorer.score(query, context)

        # 2. Self-consistency uncertainty
        sc_uncertainty = self._measure_self_consistency(query, context)

        # 3. Make routing decision
        threshold = self.thresholds.get(complexity_score.erotetic_type.value, 0.5)

        # IMPROVED: High uncertainty should trigger remote routing
        if sc_uncertainty > 0.8:
            route = "remote"
            confidence = sc_uncertainty
            rationale = f"High uncertainty ({sc_uncertainty:.2f}) triggers remote routing for safety"
        else:
            combined_score = (
                0.6 * complexity_score.overall +
                0.4 * sc_uncertainty
            )

            if combined_score < threshold:
                route = "local"
                confidence = (threshold - combined_score) / threshold
                rationale = f"Low complexity ({complexity_score.overall:.2f}) and moderate uncertainty"
            else:
                route = "remote"
                confidence = (combined_score - threshold) / (1.0 - threshold)
                rationale = f"High complexity ({complexity_score.overall:.2f}) requires remote"

        decision_time = time.time() - start_time

        return {
            "complexity_score": complexity_score.overall,
            "erotetic_type": complexity_score.erotetic_type.value,
            "bloom_level": f"{complexity_score.bloom_level.name} (L{complexity_score.bloom_level.value})",
            "reasoning_depth": complexity_score.reasoning_depth,
            "self_consistency_uncertainty": sc_uncertainty,
            "predicted_route": route,
            "confidence": confidence,
            "rationale": rationale,
            "decision_time": decision_time
        }

    def evaluate(self, ground_truth_data: Dict[str, Any]) -> List[RouterEvaluation]:
        """Evaluate router against ground truth"""
        results = []

        for i, gt_label in enumerate(ground_truth_data["ground_truth_labels"], 1):
            print(f"\n{'='*80}")
            print(f"Query {i}/{len(ground_truth_data['ground_truth_labels'])}: {gt_label['query'][:60]}...")
            print(f"{'='*80}")

            # Get router prediction
            context = None  # Ground truth data doesn't include context
            prediction = self.predict_route(gt_label["query"], context)

            print(f"Ground Truth: {gt_label['ground_truth_route'].upper()}")
            print(f"  Rationale: {gt_label['rationale']}")
            print(f"Router Prediction: {prediction['predicted_route'].upper()}")
            print(f"  Complexity: {prediction['complexity_score']:.3f} ({prediction['erotetic_type']})")
            print(f"  SC Uncertainty: {prediction['self_consistency_uncertainty']:.3f}")
            print(f"  Confidence: {prediction['confidence']:.2%}")
            print(f"  Rationale: {prediction['rationale']}")

            # Evaluate
            correct = (prediction["predicted_route"] == gt_label["ground_truth_route"])
            print(f"Result: {'✓ CORRECT' if correct else '✗ WRONG'}")

            eval_result = RouterEvaluation(
                query=gt_label["query"],
                query_type=gt_label["query_type"],
                ground_truth_route=gt_label["ground_truth_route"],
                ground_truth_rationale=gt_label["rationale"],
                complexity_score=prediction["complexity_score"],
                erotetic_type=prediction["erotetic_type"],
                bloom_level=prediction["bloom_level"],
                reasoning_depth=prediction["reasoning_depth"],
                self_consistency_uncertainty=prediction["self_consistency_uncertainty"],
                predicted_route=prediction["predicted_route"],
                router_confidence=prediction["confidence"],
                correct=correct,
                speedup_if_correct=gt_label["speedup_if_local"] if correct and prediction["predicted_route"] == "local" else 0.0,
                quality_preserved=gt_label["quality_preserved"]
            )

            results.append(eval_result)

        return results


def analyze_results(results: List[RouterEvaluation]) -> Dict[str, Any]:
    """Comprehensive analysis of router performance"""

    print(f"\n{'='*80}")
    print("ROUTER EVALUATION RESULTS")
    print(f"{'='*80}")

    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / total if total > 0 else 0

    print(f"\n📊 OVERALL ACCURACY")
    print(f"Correct: {correct}/{total} ({accuracy*100:.1f}%)")

    # Breakdown by ground truth route
    gt_local = [r for r in results if r.ground_truth_route == "local"]
    gt_remote = [r for r in results if r.ground_truth_route == "remote"]

    print(f"\n📊 ACCURACY BY GROUND TRUTH")
    if gt_local:
        correct_local = sum(1 for r in gt_local if r.correct)
        print(f"Should be LOCAL: {correct_local}/{len(gt_local)} ({correct_local/len(gt_local)*100:.1f}%)")
    if gt_remote:
        correct_remote = sum(1 for r in gt_remote if r.correct)
        print(f"Should be REMOTE: {correct_remote}/{len(gt_remote)} ({correct_remote/len(gt_remote)*100:.1f}%)")

    # Breakdown by query type
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_type[r.query_type]["total"] += 1
        if r.correct:
            by_type[r.query_type]["correct"] += 1

    print(f"\n📊 ACCURACY BY QUERY TYPE")
    for qtype in sorted(by_type.keys()):
        correct_ct = by_type[qtype]["correct"]
        total_ct = by_type[qtype]["total"]
        acc = correct_ct / total_ct if total_ct > 0 else 0
        print(f"  {qtype:12s}: {correct_ct}/{total_ct} ({acc*100:.1f}%)")

    # Speedup analysis (for correct local routes)
    correct_local_routes = [r for r in results if r.correct and r.predicted_route == "local"]
    if correct_local_routes:
        avg_speedup = np.mean([r.speedup_if_correct for r in correct_local_routes])
        print(f"\n⚡ EFFICIENCY (Correct Local Routes)")
        print(f"Queries: {len(correct_local_routes)}")
        print(f"Avg speedup: {avg_speedup:.1f}x")

    # Error analysis
    errors = [r for r in results if not r.correct]
    if errors:
        print(f"\n❌ ERROR ANALYSIS ({len(errors)} errors)")
        false_local = [r for r in errors if r.predicted_route == "local"]
        false_remote = [r for r in errors if r.predicted_route == "remote"]

        if false_local:
            print(f"\nFalse LOCAL ({len(false_local)} queries):")
            print("  Router said LOCAL but should be REMOTE (quality loss risk!)")
            for r in false_local[:3]:  # Show first 3
                print(f"    - {r.query[:60]}...")
                print(f"      Complexity: {r.complexity_score:.3f}, Uncertainty: {r.self_consistency_uncertainty:.3f}")

        if false_remote:
            print(f"\nFalse REMOTE ({len(false_remote)} queries):")
            print("  Router said REMOTE but could use LOCAL (efficiency loss)")
            for r in false_remote[:3]:
                print(f"    - {r.query[:60]}...")
                print(f"      Complexity: {r.complexity_score:.3f}, Uncertainty: {r.self_consistency_uncertainty:.3f}")

    return {
        "total_queries": total,
        "accuracy": accuracy,
        "correct_predictions": correct,
        "accuracy_by_type": {k: v["correct"]/v["total"] for k, v in by_type.items()},
        "avg_speedup": float(np.mean([r.speedup_if_correct for r in correct_local_routes])) if correct_local_routes else 0
    }


def create_visualizations(results: List[RouterEvaluation], analysis: Dict[str, Any]):
    """Create visualization of router performance"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Accuracy by query type
    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        by_type[r.query_type]["total"] += 1
        if r.correct:
            by_type[r.query_type]["correct"] += 1

    types = sorted(by_type.keys())
    accuracies = [by_type[t]["correct"]/by_type[t]["total"]*100 for t in types]

    bars = ax1.bar(types, accuracies, color='#3498db', edgecolor='black', linewidth=1.5)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Random (50%)')
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Routing Accuracy by Query Type', fontsize=13, fontweight='bold')
    ax1.set_xticklabels(types, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Confusion matrix
    tp = sum(1 for r in results if r.ground_truth_route == "local" and r.predicted_route == "local")
    fp = sum(1 for r in results if r.ground_truth_route == "remote" and r.predicted_route == "local")
    fn = sum(1 for r in results if r.ground_truth_route == "local" and r.predicted_route == "remote")
    tn = sum(1 for r in results if r.ground_truth_route == "remote" and r.predicted_route == "remote")

    confusion = np.array([[tp, fn], [fp, tn]])
    im = ax2.imshow(confusion, cmap='Blues')

    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Predicted\nLocal', 'Predicted\nRemote'])
    ax2.set_yticklabels(['Actual\nLocal', 'Actual\nRemote'])
    ax2.set_title('Confusion Matrix', fontsize=13, fontweight='bold')

    for i in range(2):
        for j in range(2):
            text = ax2.text(j, i, confusion[i, j],
                           ha="center", va="center", color="black", fontsize=20, fontweight='bold')

    # 3. Complexity score distribution by ground truth
    local_complexities = [r.complexity_score for r in results if r.ground_truth_route == "local"]
    remote_complexities = [r.complexity_score for r in results if r.ground_truth_route == "remote"]

    ax3.hist(local_complexities, bins=15, alpha=0.6, label='GT: Local', color='green')
    ax3.hist(remote_complexities, bins=15, alpha=0.6, label='GT: Remote', color='red')
    ax3.set_xlabel('Complexity Score', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Complexity Distribution by Ground Truth', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4. Summary metrics
    ax4.axis('off')
    summary_text = f"""
    ROUTER PERFORMANCE SUMMARY

    Total Queries: {analysis['total_queries']}

    Overall Accuracy: {analysis['accuracy']*100:.1f}%
    Correct: {analysis['correct_predictions']}/{analysis['total_queries']}

    True Positives (Local): {tp}
    True Negatives (Remote): {tn}
    False Positives (Local): {fp} ⚠️
    False Negatives (Remote): {fn}

    Avg Speedup (Correct Local): {analysis['avg_speedup']:.1f}x

    Key Findings:
    - False Positives = Quality risk
    - False Negatives = Efficiency loss
    - High accuracy = Good routing
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Smart Router Evaluation Against Empirical Ground Truth',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig('router_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: router_evaluation.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate router against ground truth")
    parser.add_argument("--ground-truth", type=str, required=True, help="Ground truth JSON file")
    parser.add_argument("--output", type=str, default="router_evaluation.json", help="Output file")

    args = parser.parse_args()

    print("="*80)
    print("ROUTER EVALUATION AGAINST EMPIRICAL GROUND TRUTH")
    print("="*80)

    # Load ground truth
    print(f"\nLoading ground truth from {args.ground_truth}...")
    with open(args.ground_truth, 'r') as f:
        ground_truth_data = json.load(f)

    print(f"✓ Loaded {len(ground_truth_data['ground_truth_labels'])} ground truth labels")

    # Initialize evaluator
    print("\nInitializing router...")
    from minions.clients import OllamaClient
    local_client = OllamaClient(model_name="llama3.2", temperature=0.0, max_tokens=4096, use_async=False)

    evaluator = RouterEvaluator(local_client, self_consistency_k=3)
    print("✓ Router ready\n")

    # Evaluate
    results = evaluator.evaluate(ground_truth_data)

    # Analyze
    analysis = analyze_results(results)

    # Visualize
    create_visualizations(results, analysis)

    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "analysis": analysis,
        "evaluations": [asdict(r) for r in results]
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {args.output}")
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
