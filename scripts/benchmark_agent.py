from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.db.models import TaskRun
from backend.app.services.performance import estimate_cost_usd, performance_warnings, sanitize_usage


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume latencia y uso de tareas de Radar Laboral.")
    parser.add_argument("--database-url", default=get_settings().database_url)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--input-cost", type=float)
    parser.add_argument("--output-cost", type=float)
    parser.add_argument("--cache-hit-cost", type=float)
    args = parser.parse_args()

    engine = create_engine(args.database_url)
    with Session(engine) as db:
        tasks = list(
            db.scalars(
                select(TaskRun)
                .where(TaskRun.status.in_({"completed", "failed", "cancelled"}))
                .order_by(TaskRun.updated_at.desc())
                .limit(max(1, args.limit))
            )
        )
    engine.dispose()

    results = []
    for task in tasks:
        usage = sanitize_usage(json.loads(task.usage_json or "{}"))
        results.append(
            {
                "task_run_id": task.id,
                "status": task.status,
                "usage": usage,
                "warnings": performance_warnings(usage),
                "estimated_cost_usd": estimate_cost_usd(
                    usage,
                    input_per_million=args.input_cost,
                    output_per_million=args.output_cost,
                    cache_hit_per_million=args.cache_hit_cost,
                ),
            }
        )

    warning_count = sum(len(item["warnings"]) for item in results)
    print(f"Tareas analizadas: {len(results)} - Advertencias: {warning_count}")
    print(json.dumps({"tasks": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
