"""Command line interface.

Run either ``python variantscape_pipeline/pipeline.py <command>`` or
``python -m variantscape_pipeline <command>``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from .config import Settings, gene_set_spec
from .pipeline import LLM_STAGES, STAGES, Pipeline, check_requirements
from .store import Store

log = logging.getLogger("variantscape_pipeline")


class UsageError(Exception):
    pass


def _setup_logging(settings: Settings, name: str, verbose: bool) -> Path:
    settings.ensure_dirs()
    log_file = settings.log_dir / f"{name}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    for noisy in ("httpx", "openai", "urllib3", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file


def _parse_stages(value: str) -> list[str]:
    stages = [s.strip() for s in value.split(",") if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown stage(s) {sorted(unknown)}; choose from {STAGES}")
    return stages


def archive_database(settings: Settings) -> Path | None:
    """Move the current database aside so a full run starts from scratch."""
    db = settings.db_path
    if not db.exists():
        return None
    archive_dir = settings.work_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / f"{db.stem}.{datetime.now().strftime('%Y%m%d-%H%M%S')}{db.suffix}"
    db.rename(target)
    for suffix in ("-wal", "-shm"):
        side = db.with_name(db.name + suffix)
        if side.exists():
            side.rename(target.with_name(target.name + suffix))
    return target


def plan_run(args, settings: Settings, store: Store, new_run_id: str) -> dict:
    """Decide run id, mode, stages and fetch window."""
    if args.resume:
        run = store.latest_unfinished_run()
        if not run:
            raise UsageError("Nothing to resume: there is no unfinished run.")
        log.info("Resuming run %s (%s mode)", run["run_id"], run["mode"])
        return {**run, "stages": run["stages"] or STAGES}

    stages = args.stages or STAGES
    if args.skip_llm:
        stages = [s for s in stages if s not in LLM_STAGES]
    today = date.today().isoformat()
    to_date = args.to_date or today
    if args.from_date:
        from_date = args.from_date
    elif args.mode == "full":
        from_date = settings.full_start_date
    elif "fetch" in stages:
        last = store.last_successful_fetch_run()
        if not last:
            raise UsageError("No previous successful run to continue from: "
                             "use '--mode full' for the first run, or give --from-date.")
        from_date = (date.fromisoformat(last["to_date"]) - timedelta(days=settings.overlap_days)).isoformat()
        log.info("Incremental run continues from %s (previous run %s ended %s, %d days overlap)",
                 from_date, last["run_id"], last["to_date"], settings.overlap_days)
    else:
        from_date = None
    has_fetch = "fetch" in stages
    return {"run_id": new_run_id, "mode": args.mode, "stages": stages,
            "from_date": from_date if has_fetch else None, "to_date": to_date if has_fetch else None}


def cmd_run(args, settings: Settings, store: Store, run_id: str) -> int:
    try:
        plan = plan_run(args, settings, store, run_id)
    except UsageError as exc:
        log.error("%s", exc)
        return 2
    problems = check_requirements(settings, plan["stages"], html=args.html)
    if problems:
        log.error("Cannot start run, fix these first:\n  - %s", "\n  - ".join(problems))
        return 2
    deploy_to = Path(args.deploy_to).resolve() if args.deploy_to else (settings.evidencedb_dir if args.deploy else None)
    pipeline = Pipeline(settings, store, plan["run_id"], limit=args.limit, refresh_reference=not args.reuse_reference)
    store.start_run(plan["run_id"], plan["mode"], plan["from_date"], plan["to_date"], plan["stages"])
    log.info("Run %s (%s): stages=%s window=%s..%s", plan["run_id"], plan["mode"], plan["stages"],
             plan["from_date"], plan["to_date"])
    try:
        for stage in plan["stages"]:
            log.info("=== Stage: %s ===", stage)
            if stage == "fetch":
                pipeline.fetch(plan["from_date"], plan["to_date"])
            elif stage == "build":
                pipeline.build(deploy_to=deploy_to, html=args.html, force=args.force)
            else:
                getattr(pipeline, stage)()
    except Exception:
        log.exception("Run %s failed; continue it with 'run --resume'", plan["run_id"])
        store.finish_run(plan["run_id"], "failed", pipeline.summary)
        return 1
    if "fetch" in plan["stages"] and pipeline.fetched_through != plan["to_date"]:
        if pipeline.fetched_through:
            log.warning("Fetch completed only through %s; the next incremental run continues from there",
                        pipeline.fetched_through)
        else:
            log.warning("No month of the window was fetched completely; this run does not count as a "
                        "starting point for incremental runs")
        store.set_run_end_date(plan["run_id"], pipeline.fetched_through)
    store.finish_run(plan["run_id"], "ok", pipeline.summary)
    log.info("Run %s finished: %s", plan["run_id"], json.dumps(pipeline.summary, default=str, indent=1))
    return 0


def cmd_status(args, settings: Settings, store: Store, run_id: str) -> int:
    print(f"Database: {settings.db_path}")
    for status, n in store.conn.execute("SELECT status, COUNT(*) FROM papers GROUP BY status"):
        print(f"  papers[{status}]: {n:,}")
    for stage, n in store.conn.execute("SELECT stage, COUNT(*) FROM stage_done GROUP BY stage"):
        print(f"  stage[{stage}] done: {n:,}")
    for table in ("gene_hits", "cancer_terms", "treatment_hits", "variant_llm", "study_design", "verify_llm"):
        print(f"  {table}: {store.scalar(f'SELECT COUNT(*) FROM {table}'):,} rows")
    print("Recent runs:")
    for row in store.conn.execute("SELECT run_id, mode, status, from_date, to_date, finished_at FROM runs "
                                  "ORDER BY started_at DESC LIMIT 5"):
        print("  ", *row)
    return 0


GENE_SET_HELP = ("Genes that make a paper relevant: 'oncology' (default; OncoKB cancer genes + CIViC genes), "
                 "'civic', or a panel file with one symbol per line / first CSV column. 'civic' and panels also "
                 "restrict graph variants to their genes. Overrides VARIANTSCAPE_GENE_SET.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="variantscape_pipeline", description=__doc__)
    parser.add_argument("--env-file", help="Path to a .env file (default: .env in the working directory)")
    parser.add_argument("--work-dir", help="Override VARIANTSCAPE_WORK_DIR")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the pipeline")
    run.add_argument("--mode", choices=["incremental", "full"], default="incremental",
                     help="incremental (default): fetch from the end of the last successful run (minus "
                          "VARIANTSCAPE_OVERLAP_DAYS) to today and process only new papers. "
                          "full: archive the database and rebuild everything from VARIANTSCAPE_FULL_START_DATE.")
    run.add_argument("--resume", action="store_true",
                     help="Continue the most recent unfinished run (same window, mode and stages)")
    run.add_argument("--from-date", help="Override the start of the publication window (YYYY-MM-DD)")
    run.add_argument("--to-date", help="Override the end of the publication window (default: today)")
    run.add_argument("--stages", type=_parse_stages, help=f"Comma-separated subset of {','.join(STAGES)}")
    run.add_argument("--skip-llm", action="store_true", help="Skip the LLM stages (variants, study_design, verify)")
    run.add_argument("--limit", type=int, help="Cap fetched works and LLM calls per stage (for testing/budgets)")
    run.add_argument("--reuse-reference", action="store_true", help="Reuse the latest CIViC snapshot instead of refreshing")
    run.add_argument("--gene-set", help=GENE_SET_HELP)
    run.add_argument("--html", action="store_true", help="Also render the interactive network HTML (slow)")
    run.add_argument("--deploy", action="store_true", help="Copy artifacts to VARIANTSCAPE_EVIDENCEDB_DIR")
    run.add_argument("--deploy-to", help="Copy artifacts into this EvidenceDb checkout")
    run.add_argument("--force", action="store_true", help="Deploy even if the graph shrank by more than 20%%")
    run.set_defaults(func=cmd_run)

    build = sub.add_parser("build", help="Only rebuild (and optionally deploy) the artifacts from stored results")
    build.add_argument("--gene-set", help=GENE_SET_HELP)
    build.add_argument("--html", action="store_true")
    build.add_argument("--deploy", action="store_true")
    build.add_argument("--deploy-to")
    build.add_argument("--force", action="store_true")
    build.add_argument("--reuse-reference", action="store_true")
    build.set_defaults(func=cmd_run, mode="build", resume=False, stages=["build"], skip_llm=False, limit=None,
                       from_date=None, to_date=None)

    status = sub.add_parser("status", help="Show database contents and recent runs")
    status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    if getattr(args, "resume", False) and (args.from_date or args.to_date or args.stages or args.mode != "incremental"):
        parser.error("--resume reuses the interrupted run's settings; it cannot be combined with "
                     "--mode, --from-date, --to-date or --stages")

    settings = Settings.from_env(args.env_file)
    if args.work_dir:
        settings.work_dir = Path(args.work_dir).resolve()
    if getattr(args, "gene_set", None):
        # A relative panel path on the command line is relative to the current directory
        spec = args.gene_set if args.gene_set.lower() in {"oncology", "civic"} else str(Path(args.gene_set).resolve())
        settings.gene_set = gene_set_spec(spec)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.command != "status":
        _setup_logging(settings, f"{args.command}-{run_id}", args.verbose)
    settings.ensure_dirs()
    if args.command == "run" and args.mode == "full" and not args.resume:
        # Check requirements before moving the database aside, so a run that cannot start keeps it
        stages = [s for s in (args.stages or STAGES) if not (args.skip_llm and s in LLM_STAGES)]
        problems = check_requirements(settings, stages, html=args.html)
        if problems:
            log.error("Cannot start run, fix these first:\n  - %s", "\n  - ".join(problems))
            return 2
        archived = archive_database(settings)
        if archived:
            log.info("Full run: previous database archived to %s", archived)
    store = Store(settings.db_path)
    try:
        return args.func(args, settings, store, run_id)
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
