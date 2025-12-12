"""
Generate Additional Test Queries for Routing Evaluation

This script generates diverse queries across different types and difficulty levels,
with ground truth answers for evaluation.

Usage:
    python experiments/generate_queries.py --count 63 --output generated_queries.json
"""

import argparse
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minions.clients import TogetherClient


# Query templates by type
QUERY_TEMPLATES = {
    "lookup": {
        "patterns": [
            "What is {term}",
            "Define {term}",
            "What does {term} mean",
            "Explain {term}",
            "What is the meaning of {term}",
            "Describe {term}",
            "What are {term}",
        ],
        "terms": [
            "photosynthesis", "mitochondria", "DNA", "ecosystem", "metabolism",
            "gravity", "momentum", "velocity", "acceleration", "friction",
            "inflation", "supply and demand", "compound interest", "recession", "monopoly",
            "democracy", "constitution", "sovereignty", "federalism", "judiciary",
            "Renaissance", "Industrial Revolution", "Cold War", "Enlightenment", "Reformation",
            "algorithm", "recursion", "encryption", "API", "database",
            "quantum mechanics", "relativity", "entropy", "wavelength", "magnetism",
        ],
        "expected_route": "local",
        "difficulty": "easy"
    },

    "math": {
        "patterns": [
            ("Calculate {a} + {b} × {c}", lambda a, b, c: a + b * c),
            ("What is {a}% of {b}", lambda a, b: a * b / 100),
            ("If {a} items cost ${b}, what's the cost per item", lambda a, b: b / a),
            ("What is {a} × {b} ÷ {c}", lambda a, b, c: a * b / c),
            ("Calculate {a}² - {b}", lambda a, b: a**2 - b),
            ("What is the average of {a}, {b}, and {c}", lambda a, b, c: (a + b + c) / 3),
            ("If you invest ${a} at {b}% interest for 1 year, what's the total", lambda a, b: a * (1 + b/100)),
        ],
        "params": {
            "a": range(5, 100, 5),
            "b": range(5, 100, 5),
            "c": range(2, 20, 2)
        },
        "expected_route": "local",
        "difficulty": "easy"
    },

    "multi-hop": {
        "patterns": [
            "How does {topic1} affect {topic2}",
            "What is the relationship between {topic1} and {topic2}",
            "Compare {topic1} and {topic2} in terms of {aspect}",
            "Analyze the impact of {topic1} on {topic2}",
            "What are the pros and cons of {topic1} versus {topic2}",
            "Evaluate {topic1} considering {aspect1} and {aspect2}",
            "Discuss how {topic1} influences {topic2} and {topic3}",
        ],
        "topics": {
            "topic1": [
                "climate change", "artificial intelligence", "globalization",
                "social media", "automation", "urbanization", "deforestation",
                "income inequality", "technological advancement", "population growth"
            ],
            "topic2": [
                "economic growth", "education systems", "healthcare access",
                "political stability", "environmental sustainability", "job markets",
                "social relationships", "mental health", "income distribution", "food security"
            ],
            "topic3": [
                "technological innovation", "cultural diversity", "resource allocation"
            ],
            "aspect": [
                "efficiency", "sustainability", "equity", "scalability", "accessibility"
            ],
            "aspect1": ["environmental impact", "economic viability", "social equity"],
            "aspect2": ["long-term sustainability", "short-term costs", "ethical implications"]
        },
        "expected_route": "remote",
        "difficulty": "hard"
    },

    "code": {
        "patterns": [
            "Write a Python function to {task}",
            "Implement {task} in Python",
            "Create a function that {task}",
            "How do I {task} in Python",
            "Write code to {task}",
        ],
        "tasks": [
            "find the factorial of a number",
            "check if a string is a palindrome",
            "sort a list using bubble sort",
            "find the GCD of two numbers",
            "convert decimal to binary",
            "calculate the Fibonacci sequence up to n terms",
            "remove duplicates from a list",
            "find the second largest number in a list",
            "count vowels in a string",
            "merge two sorted lists",
            "implement a stack using a list",
            "check if a number is an Armstrong number",
            "find the sum of digits of a number",
            "reverse words in a sentence",
            "find all prime numbers up to n",
        ],
        "expected_route": "local",
        "difficulty": "medium"
    },

    "extract": {
        "patterns": [
            "Based on the context, {question}",
            "According to the information provided, {question}",
            "From the given text, {question}",
            "Using the context, {question}",
        ],
        "questions": [
            "what is the main idea",
            "who are the key people mentioned",
            "what are the important dates",
            "what is the conclusion",
            "what are the supporting arguments",
            "what evidence is provided",
            "what is the author's position",
        ],
        "expected_route": "local",
        "difficulty": "medium"
    },

    "open-ended": {
        "patterns": [
            "What are your thoughts on {topic}",
            "Should society {action}",
            "Is {statement} a good or bad thing",
            "What is the best approach to {problem}",
            "How can we improve {area}",
            "What would happen if {scenario}",
        ],
        "topics": [
            "universal basic income",
            "genetic engineering in humans",
            "colonizing Mars",
            "mandatory voting",
            "nuclear energy",
            "surveillance for public safety",
        ],
        "actions": [
            "ban single-use plastics",
            "implement a carbon tax",
            "require AI ethics training",
            "prioritize renewable energy",
        ],
        "statements": [
            "social media regulation",
            "remote work becoming the norm",
            "increasing automation in workplaces",
        ],
        "problems": [
            "climate change",
            "healthcare access",
            "education inequality",
        ],
        "areas": [
            "public transportation",
            "mental health support",
            "renewable energy adoption",
        ],
        "scenarios": [
            "all cars became electric",
            "AI could do most human jobs",
            "we achieved fusion energy",
        ],
        "expected_route": "remote",
        "difficulty": "hard"
    }
}


class QueryGenerator:
    """Generate diverse test queries with ground truth"""

    def __init__(self, llm_client, existing_queries=None):
        self.llm = llm_client
        self.existing_queries = set()

        # Load existing queries to avoid duplicates
        if existing_queries:
            for q in existing_queries:
                self.existing_queries.add(q.lower().strip())

    def is_duplicate(self, query: str) -> bool:
        """Check if query already exists"""
        return query.lower().strip() in self.existing_queries

    def generate_lookup_queries(self, count: int) -> List[Dict[str, Any]]:
        """Generate factual lookup queries"""
        queries = []
        templates = QUERY_TEMPLATES["lookup"]

        import random
        random.seed(42)

        attempts = 0
        max_attempts = count * 10  # Allow some retries

        while len(queries) < count and attempts < max_attempts:
            attempts += 1

            pattern = random.choice(templates["patterns"])
            term = random.choice(templates["terms"])
            query_text = pattern.format(term=term)

            # Skip duplicates
            if self.is_duplicate(query_text):
                continue

            # Generate ground truth using LLM
            prompt = f"""Provide a concise, factual answer (2-3 sentences) to: {query_text}

Answer:"""
            result = self.llm.chat([{"role": "user", "content": prompt}])
            ground_truth = result[0][0] if isinstance(result[0], list) else result[0]

            queries.append({
                "query": query_text,
                "query_type": "lookup",
                "expected_route": "local",
                "ground_truth": ground_truth.strip(),
                "difficulty": "easy",
                "context": []
            })

            # Add to existing set
            self.existing_queries.add(query_text.lower().strip())

            print(f"Generated lookup {len(queries)}/{count}: {query_text[:50]}...")

        if len(queries) < count:
            print(f"Warning: Only generated {len(queries)}/{count} lookup queries (rest were duplicates)")

        return queries

    def generate_math_queries(self, count: int) -> List[Dict[str, Any]]:
        """Generate math queries with computed answers"""
        queries = []
        templates = QUERY_TEMPLATES["math"]

        import random
        random.seed(43)

        for i in range(count):
            pattern_data = random.choice(templates["patterns"])
            if isinstance(pattern_data, tuple):
                pattern, compute = pattern_data
            else:
                continue

            # Generate random parameters
            a = random.choice(list(templates["params"]["a"]))
            b = random.choice(list(templates["params"]["b"]))
            c = random.choice(list(templates["params"]["c"]))

            # Create query and compute answer
            query_text = pattern.format(a=a, b=b, c=c)

            try:
                # Compute ground truth
                if pattern.count("{") == 2:
                    answer = compute(a, b)
                else:
                    answer = compute(a, b, c)

                ground_truth = f"The answer is {answer:.2f}" if isinstance(answer, float) else f"The answer is {answer}"

                queries.append({
                    "query": query_text,
                    "query_type": "math",
                    "expected_route": "local",
                    "ground_truth": ground_truth,
                    "difficulty": "easy",
                    "context": []
                })

                print(f"Generated math {i+1}/{count}: {query_text[:50]}...")
            except Exception as e:
                print(f"Error generating math query: {e}")
                continue

        return queries

    def generate_multihop_queries(self, count: int) -> List[Dict[str, Any]]:
        """Generate multi-hop reasoning queries"""
        queries = []
        templates = QUERY_TEMPLATES["multi-hop"]

        import random
        random.seed(44)

        for i in range(count):
            pattern = random.choice(templates["patterns"])

            # Fill in template with random topics
            topic_vars = {k: random.choice(v) for k, v in templates["topics"].items() if f"{{{k}}}" in pattern}
            query_text = pattern.format(**topic_vars)

            # Generate ground truth using LLM
            prompt = f"""Provide a comprehensive answer (4-6 sentences) analyzing: {query_text}

Include multiple perspectives and key considerations.

Answer:"""
            result = self.llm.chat([{"role": "user", "content": prompt}])
            ground_truth = result[0][0] if isinstance(result[0], list) else result[0]

            queries.append({
                "query": query_text,
                "query_type": "multi-hop",
                "expected_route": "remote",
                "ground_truth": ground_truth.strip(),
                "difficulty": "hard",
                "context": []
            })

            print(f"Generated multi-hop {i+1}/{count}: {query_text[:50]}...")

        return queries

    def generate_code_queries(self, count: int) -> List[Dict[str, Any]]:
        """Generate coding task queries"""
        queries = []
        templates = QUERY_TEMPLATES["code"]

        import random
        random.seed(45)

        for i in range(count):
            pattern = random.choice(templates["patterns"])
            task = random.choice(templates["tasks"])
            query_text = pattern.format(task=task)

            # Generate ground truth code using LLM
            prompt = f"""{query_text}

Provide a complete, working Python function.

Code:"""
            result = self.llm.chat([{"role": "user", "content": prompt}])
            ground_truth = result[0][0] if isinstance(result[0], list) else result[0]

            queries.append({
                "query": query_text,
                "query_type": "code",
                "expected_route": "local",
                "ground_truth": ground_truth.strip(),
                "difficulty": "medium",
                "context": []
            })

            print(f"Generated code {i+1}/{count}: {query_text[:50]}...")

        return queries

    def generate_openended_queries(self, count: int) -> List[Dict[str, Any]]:
        """Generate open-ended discussion queries"""
        queries = []
        templates = QUERY_TEMPLATES["open-ended"]

        import random
        random.seed(46)

        # Mapping from plural keys to singular placeholders
        key_mapping = {
            "topics": "topic",
            "actions": "action",
            "statements": "statement",
            "problems": "problem",
            "areas": "area",
            "scenarios": "scenario"
        }

        for i in range(count):
            pattern = random.choice(templates["patterns"])

            # Fill in template with proper key mapping
            topic_vars = {}
            for k, v in templates.items():
                if k in key_mapping:
                    placeholder = key_mapping[k]
                    if f"{{{placeholder}}}" in pattern:
                        topic_vars[placeholder] = random.choice(v)

            query_text = pattern.format(**topic_vars)

            # Generate ground truth using LLM
            prompt = f"""Provide a balanced, thoughtful response to: {query_text}

Include multiple perspectives, key arguments, and considerations.

Answer:"""
            result = self.llm.chat([{"role": "user", "content": prompt}])
            ground_truth = result[0][0] if isinstance(result[0], list) else result[0]

            queries.append({
                "query": query_text,
                "query_type": "open-ended",
                "expected_route": "remote",
                "ground_truth": ground_truth.strip(),
                "difficulty": "hard",
                "context": []
            })

            print(f"Generated open-ended {i+1}/{count}: {query_text[:50]}...")

        return queries


def main():
    parser = argparse.ArgumentParser(description="Generate test queries")
    parser.add_argument("--count", type=int, default=63, help="Number of additional queries to generate")
    parser.add_argument("--output", type=str, default="generated_queries.json", help="Output file")
    parser.add_argument("--existing", type=str, default=None,
                       help="Existing queries file to avoid duplicates (JSON or from test_data.py)")
    parser.add_argument("--distribution", type=str, default="balanced",
                       choices=["balanced", "custom"],
                       help="Distribution of query types")

    args = parser.parse_args()

    print("="*80)
    print("QUERY GENERATION")
    print("="*80)
    print(f"\nGenerating {args.count} queries...")

    # Load existing queries to avoid duplicates
    existing_query_texts = []

    if args.existing:
        print(f"\nLoading existing queries from {args.existing} to avoid duplicates...")
        try:
            with open(args.existing, 'r') as f:
                existing_data = json.load(f)
                existing_queries = existing_data.get("queries", [])
                existing_query_texts = [q["query"] for q in existing_queries]
                print(f"✓ Loaded {len(existing_query_texts)} existing queries")
        except:
            print(f"Warning: Could not load {args.existing}, continuing without deduplication")

    # Also load from test_data.py
    try:
        from experiments.test_data import TestDataset
        dataset = TestDataset()
        test_queries = dataset.get_all()
        existing_query_texts.extend([q.query for q in test_queries])
        print(f"✓ Loaded {len(test_queries)} queries from test_data.py")
    except:
        print("Warning: Could not load test_data.py")

    print(f"\nTotal existing queries to avoid: {len(existing_query_texts)}")

    # Initialize LLM for ground truth generation
    print("\nInitializing LLM client for ground truth generation...")
    llm_client = TogetherClient(model="Qwen/Qwen2.5-72B-Instruct-Turbo")
    print("✓ Client ready\n")

    generator = QueryGenerator(llm_client, existing_queries=existing_query_texts)

    # Distribute queries across types
    if args.distribution == "balanced":
        per_type = args.count // 6
        remainder = args.count % 6

        counts = {
            "lookup": per_type + (1 if remainder > 0 else 0),
            "math": per_type + (1 if remainder > 1 else 0),
            "multi-hop": per_type + (1 if remainder > 2 else 0),
            "code": per_type + (1 if remainder > 3 else 0),
            "open-ended": per_type + (1 if remainder > 4 else 0),
        }

    print("Distribution:")
    for qtype, count in counts.items():
        print(f"  {qtype}: {count}")
    print()

    # Generate queries
    all_queries = []

    print("Generating lookup queries...")
    all_queries.extend(generator.generate_lookup_queries(counts["lookup"]))

    print("\nGenerating math queries...")
    all_queries.extend(generator.generate_math_queries(counts["math"]))

    print("\nGenerating multi-hop queries...")
    all_queries.extend(generator.generate_multihop_queries(counts["multi-hop"]))

    print("\nGenerating code queries...")
    all_queries.extend(generator.generate_code_queries(counts["code"]))

    print("\nGenerating open-ended queries...")
    all_queries.extend(generator.generate_openended_queries(counts["open-ended"]))

    # Save
    output_data = {
        "metadata": {
            "total_queries": len(all_queries),
            "distribution": counts,
            "generation_method": "LLM-assisted with templates"
        },
        "queries": all_queries
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ Generated {len(all_queries)} queries")
    print(f"✓ Saved to {args.output}")
    print(f"{'='*80}")

    # Statistics
    by_type = {}
    by_route = {}
    for q in all_queries:
        qtype = q["query_type"]
        route = q["expected_route"]
        by_type[qtype] = by_type.get(qtype, 0) + 1
        by_route[route] = by_route.get(route, 0) + 1

    print("\nStatistics:")
    print(f"  Total: {len(all_queries)}")
    print(f"  By type: {by_type}")
    print(f"  By expected route: {by_route}")
    print(f"  Local: {by_route.get('local', 0)} ({by_route.get('local', 0)/len(all_queries)*100:.1f}%)")
    print(f"  Remote: {by_route.get('remote', 0)} ({by_route.get('remote', 0)/len(all_queries)*100:.1f}%)")


if __name__ == "__main__":
    main()
