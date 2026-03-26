"""
Benchmark: pgvector vs Databricks Vector Search for MaintBot memory.

Seeds both backends with identical test memories, runs a set of queries,
and compares latency, result quality, and ranking consistency.

Usage:
    uv run python scripts/benchmark_memory.py

Requires:
    - Lakebase (pgvector) configured in .env
    - Vector Search endpoint + index provisioned (run setup_vector_search.py first)
"""

import asyncio
import logging
import os
import statistics
import time
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Test data
# ──────────────────────────────────────────────

TEST_TECHNICIAN_ID = "benchmark-tech-001"

TEST_MEMORIES = [
    {"content": "HP-L4-001 hydraulic press has a recurring seal leak on the left cylinder, typically after 2000 cycles", "memory_type": "failure_mode", "equipment_tag": "HP-L4-001", "importance": 0.9},
    {"content": "Always use Parker 2-012 N674-70 seals for HP-L4 series hydraulic presses, the generic ones fail within 500 cycles", "memory_type": "part_preference", "equipment_tag": "HP-L4-001", "importance": 0.8},
    {"content": "When diagnosing CNC-500X spindle noise, check bearing preload first before replacing the spindle unit", "memory_type": "procedure", "equipment_tag": "CNC-500X", "importance": 0.85},
    {"content": "CNC-500X firmware v3.2.1 has a known bug causing false temperature alarms, update to v3.2.3", "memory_type": "machine_quirk", "equipment_tag": "CNC-500X", "importance": 0.7},
    {"content": "Parker Hannifin lead time for custom hydraulic seals is currently 6-8 weeks, order from Midwest Hydraulics for 2-week delivery", "memory_type": "vendor_info", "equipment_tag": None, "importance": 0.6},
    {"content": "Conveyor belt CB-M2-003 alignment drifts every 72 hours due to worn tensioner pulley bearing", "memory_type": "failure_mode", "equipment_tag": "CB-M2-003", "importance": 0.75},
    {"content": "For electrical troubleshooting on WR-E1-002 welding robot, always check the ground connection at junction box J4 first", "memory_type": "procedure", "equipment_tag": "WR-E1-002", "importance": 0.8},
    {"content": "The pneumatic actuator on AS-P3-001 assembly station requires exactly 85 PSI, anything above 90 damages the end effector", "memory_type": "machine_quirk", "equipment_tag": "AS-P3-001", "importance": 0.95},
    {"content": "SKF 6205-2RS bearings are the standard replacement for all Line 4 conveyor idler rollers", "memory_type": "part_preference", "equipment_tag": "CB-M2-003", "importance": 0.65},
    {"content": "After replacing the servo drive on WR-E1-002, the robot needs a full kinematic recalibration, takes about 45 minutes", "memory_type": "procedure", "equipment_tag": "WR-E1-002", "importance": 0.7},
    {"content": "HP-L4-001 oil should be changed every 3000 hours, use Mobil DTE 25 hydraulic oil only", "memory_type": "procedure", "equipment_tag": "HP-L4-001", "importance": 0.8},
    {"content": "The thermal camera showed hotspot on CNC-500X spindle motor at 185F during heavy cuts, normal is under 160F", "memory_type": "failure_mode", "equipment_tag": "CNC-500X", "importance": 0.85},
]

TEST_QUERIES = [
    "HP-L4-001 is leaking hydraulic fluid, what do we know about this?",
    "CNC spindle making noise during operation",
    "What parts should I order for the hydraulic press?",
    "Conveyor belt keeps going out of alignment",
    "Welding robot electrical problems",
    "What's the correct pressure for the pneumatic assembly station?",
    "Need to do maintenance on the hydraulic press, any tips?",
    "Temperature issues with CNC machine",
]


# ──────────────────────────────────────────────
# Seed functions
# ──────────────────────────────────────────────

async def seed_pgvector(tech_id: str):
    """Seed pgvector backend with test memories."""
    from agent_server.memory import get_embedding, store_memory

    logger.info(f"Seeding pgvector with {len(TEST_MEMORIES)} memories...")
    for mem in TEST_MEMORIES:
        await store_memory(
            technician_id=tech_id,
            content=mem["content"],
            memory_type=mem["memory_type"],
            equipment_tag=mem.get("equipment_tag"),
            importance=mem["importance"],
            source_work_order_id=None,
        )
    logger.info("pgvector seeded")


async def seed_vector_search(tech_id: str):
    """Seed Vector Search backend with test memories."""
    from agent_server.memory_vs import store_memory_vs

    logger.info(f"Seeding Vector Search with {len(TEST_MEMORIES)} memories...")
    for mem in TEST_MEMORIES:
        await store_memory_vs(
            technician_id=tech_id,
            content=mem["content"],
            memory_type=mem["memory_type"],
            equipment_tag=mem.get("equipment_tag"),
            importance=mem["importance"],
            source_work_order_id=None,
        )
    logger.info("Vector Search seeded")


# ──────────────────────────────────────────────
# Benchmark functions
# ──────────────────────────────────────────────

async def benchmark_backend(
    backend_name: str,
    retrieve_fn,
    tech_id: str,
    queries: List[str],
    runs_per_query: int = 3,
) -> Dict[str, Any]:
    """Run queries against a backend and measure latency + results."""
    all_latencies = []
    query_results = {}

    for query in queries:
        latencies = []
        results = None
        for _ in range(runs_per_query):
            t0 = time.perf_counter()
            results = await retrieve_fn(tech_id, query, limit=5)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

        all_latencies.extend(latencies)
        query_results[query] = {
            "latencies_ms": [round(l, 1) for l in latencies],
            "avg_ms": round(statistics.mean(latencies), 1),
            "results": [
                {"content": r.get("content", "")[:80], "score": r.get("final_score", 0)}
                for r in (results or [])
            ],
            "result_count": len(results or []),
        }

    return {
        "backend": backend_name,
        "total_queries": len(queries) * runs_per_query,
        "p50_ms": round(statistics.median(all_latencies), 1),
        "p95_ms": round(sorted(all_latencies)[int(len(all_latencies) * 0.95)], 1) if len(all_latencies) >= 20 else round(max(all_latencies), 1),
        "p99_ms": round(sorted(all_latencies)[int(len(all_latencies) * 0.99)], 1) if len(all_latencies) >= 100 else round(max(all_latencies), 1),
        "avg_ms": round(statistics.mean(all_latencies), 1),
        "min_ms": round(min(all_latencies), 1),
        "max_ms": round(max(all_latencies), 1),
        "queries": query_results,
    }


def compute_overlap(pg_results: Dict, vs_results: Dict) -> Dict[str, Any]:
    """Compare result overlap between backends for each query."""
    overlaps = {}
    for query in pg_results["queries"]:
        pg_contents = {r["content"] for r in pg_results["queries"][query]["results"]}
        vs_contents = {r["content"] for r in vs_results["queries"].get(query, {}).get("results", [])}
        common = pg_contents & vs_contents
        total = pg_contents | vs_contents
        overlaps[query[:60]] = {
            "pgvector_count": len(pg_contents),
            "vs_count": len(vs_contents),
            "overlap": len(common),
            "jaccard": round(len(common) / len(total), 2) if total else 0,
        }
    return overlaps


def print_report(pg_stats: Dict, vs_stats: Dict, overlaps: Dict):
    """Print a comparison report."""
    print("\n" + "=" * 80)
    print("MEMORY BACKEND BENCHMARK: pgvector vs Databricks Vector Search")
    print("=" * 80)

    print(f"\n{'Metric':<25} {'pgvector':>15} {'Vector Search':>15} {'Difference':>15}")
    print("-" * 70)
    for metric in ["avg_ms", "p50_ms", "p95_ms", "min_ms", "max_ms"]:
        pg_val = pg_stats[metric]
        vs_val = vs_stats[metric]
        diff = vs_val - pg_val
        sign = "+" if diff > 0 else ""
        print(f"{metric:<25} {pg_val:>12.1f}ms {vs_val:>12.1f}ms {sign}{diff:>12.1f}ms")

    print(f"\n{'Total queries':<25} {pg_stats['total_queries']:>15} {vs_stats['total_queries']:>15}")

    print("\n--- Result Overlap (Jaccard similarity) ---")
    print(f"{'Query':<62} {'Overlap':>8} {'Jaccard':>8}")
    print("-" * 78)
    for query, overlap in overlaps.items():
        print(f"{query:<62} {overlap['overlap']:>5}/{max(overlap['pgvector_count'], overlap['vs_count'])} {overlap['jaccard']:>7.0%}")

    print("\n--- Per-Query Latency ---")
    print(f"{'Query':<50} {'pgvector':>12} {'VS':>12}")
    print("-" * 74)
    for query in pg_stats["queries"]:
        pg_avg = pg_stats["queries"][query]["avg_ms"]
        vs_avg = vs_stats["queries"].get(query, {}).get("avg_ms", 0)
        print(f"{query[:50]:<50} {pg_avg:>9.1f}ms {vs_avg:>9.1f}ms")

    print("\n" + "=" * 80)

    # Winner summary
    pg_faster = pg_stats["avg_ms"] < vs_stats["avg_ms"]
    avg_jaccard = statistics.mean(o["jaccard"] for o in overlaps.values()) if overlaps else 0
    print(f"\nLatency winner: {'pgvector' if pg_faster else 'Vector Search'} "
          f"({abs(pg_stats['avg_ms'] - vs_stats['avg_ms']):.1f}ms faster on average)")
    print(f"Average result overlap (Jaccard): {avg_jaccard:.0%}")
    print(f"  > 80% = highly similar results, backends are interchangeable")
    print(f"  < 50% = significant differences, investigate ranking")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

async def run_benchmark():
    """Main benchmark workflow."""
    # Resolve or create benchmark technician
    from agent_server.db_schema import ensure_schema, get_or_create_technician
    await ensure_schema()
    tech = await get_or_create_technician(TEST_TECHNICIAN_ID)
    tech_id = str(tech["id"])
    logger.info(f"Benchmark technician: {TEST_TECHNICIAN_ID} -> {tech_id}")

    # Seed both backends
    await seed_pgvector(tech_id)
    await seed_vector_search(tech_id)

    # Import retrieval functions
    from agent_server.memory import retrieve_memories_hybrid as pg_retrieve
    from agent_server.memory_vs import retrieve_memories_vs as vs_retrieve

    # Warm-up run (1 query each to warm caches/connections)
    logger.info("Warm-up run...")
    await pg_retrieve(tech_id, "test query", limit=3)
    await vs_retrieve(tech_id, "test query", limit=3)

    # Benchmark
    logger.info("Running pgvector benchmark...")
    pg_stats = await benchmark_backend("pgvector", pg_retrieve, tech_id, TEST_QUERIES, runs_per_query=3)

    logger.info("Running Vector Search benchmark...")
    vs_stats = await benchmark_backend("databricks_vs", vs_retrieve, tech_id, TEST_QUERIES, runs_per_query=3)

    # Compare
    overlaps = compute_overlap(pg_stats, vs_stats)
    print_report(pg_stats, vs_stats, overlaps)


def main():
    asyncio.run(run_benchmark())


if __name__ == "__main__":
    main()
