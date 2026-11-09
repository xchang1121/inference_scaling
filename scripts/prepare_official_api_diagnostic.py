"""Prepare a small, manually authored official-document rule diagnostic.

This is not a published benchmark. No service API is called. The eight short
corpus entries paraphrase official documentation; raw page snapshots and hashes
are kept separately for audit. Questions and answer labels never enter the index.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


SOURCES = [
    ("stripe", "Stripe API v1 and v2 idempotent requests",
     "https://docs.stripe.com/api/idempotent_requests",
     "API v1 saves the first executed request's status and response body for an idempotency key, including HTTP 500. Repeating the same key and parameters replays that saved result. In API v2, a retry can finish failed or partially completed work and return an updated response; successfully completed operations are not repeated. These are different retry contracts."),
    ("redis", "Redis EXPIRE conditional expiry GT NX XX",
     "https://redis.io/docs/latest/commands/expire/",
     "EXPIRE GT changes a key's expiry only if the requested lifetime exceeds the current remaining lifetime. A persistent key has infinite lifetime for this comparison, so a finite GT request leaves it persistent. NX requires no existing expiry; XX requires an existing expiry. The conditions are mutually exclusive."),
    ("postgresql", "PostgreSQL 16 transaction isolation and serialization failures",
     "https://www.postgresql.org/docs/16/transaction-iso.html",
     "At Read Committed, each ordinary SELECT uses the committed database snapshot at that statement's start; successive SELECT statements in one transaction can therefore see another transaction's newly committed changes. Applications must handle serialization failures in Repeatable Read or Serializable transactions by aborting and retrying the whole transaction, including its earlier reads."),
    ("sqs", "Amazon SQS visibility timeout and duplicate delivery",
     "https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-visibility-timeout.html",
     "Receiving an SQS message does not delete it. Its visibility timer begins at delivery. If it is not deleted, it becomes visible for another receive when that timer expires. ChangeMessageVisibility adjusts the timer. Because delivery is at least once, the visibility interval does not provide an absolute promise against duplicate delivery, including for a standard queue."),
    ("dict", "Python 3.13 dict get and setdefault",
     "https://docs.python.org/3.13/library/stdtypes.html#dict.setdefault",
     "For a missing dictionary key, dict.get(key, default) returns the default and does not add an entry. dict.setdefault(key, default) adds an entry with that default and returns it. For an existing key, both methods return its current value and setdefault leaves that value unchanged."),
    ("lru", "Python 3.13 functools lru_cache concurrency and references",
     "https://docs.python.org/3.13/library/functools.html#functools.lru_cache",
     "functools.lru_cache keeps its internal cache coherent across threads, but this is not single-flight execution: simultaneous first misses for the same arguments can run the wrapped function more than once before a result is stored. Cache entries hold references to their arguments and returned objects until eviction or an explicit cache_clear."),
    ("asyncio", "Python 3.13 asyncio wait_for cancellation and shield",
     "https://docs.python.org/3.13/library/asyncio-task.html#asyncio.wait_for",
     "asyncio.wait_for cancels its awaited task when the timeout expires, and waits for cancellation to finish before completing. Thus the total elapsed time may be longer than the timeout. Passing shield(task) protects the underlying task against this cancellation, although the caller of wait_for still receives the timeout exception."),
    ("http", "RFC 9110 section 9.2.2 idempotent method retries",
     "https://www.rfc-editor.org/rfc/rfc9110.html#section-9.2.2",
     "PUT, DELETE, and safe HTTP methods are idempotent in their intended server effect. After a connection failure before a response is read, a client may automatically retry an idempotent request. A client should avoid automatically retrying a non-idempotent method unless it knows the actual operation is idempotent or knows the first request was never applied. Repeated responses need not be identical."),
]

# Each tuple contains a scenario, the allowed decision labels, the correct
# decision, and a mandatory waiting time. Decisions are labels, not API calls.
CASES = {
    "stripe": [
        ("Stripe API v1 began endpoint execution, then returned HTTP 500. The client repeats the same unpruned idempotency key and unchanged parameters. Which response contract applies?",
         ["return_saved_500", "execute_again", "reject_key"], "return_saved_500", 0),
        ("Stripe API v2 returned a partial failure. On retry, the remaining failed work can now succeed. Which behavior follows the v2 idempotency contract?",
         ["replay_original_500", "retry_incomplete_work", "repeat_all_successful_work"], "retry_incomplete_work", 0),
    ],
    "redis": [
        ("In Redis, an existing key has 20 seconds of TTL remaining. EXPIRE key 60 GT is issued immediately. What should happen?",
         ["set_expiry", "keep_existing_ttl", "delete_key"], "set_expiry", 0),
        ("In Redis, an existing key is persistent and has no expiry. EXPIRE key 60 GT is issued. What should happen?",
         ["set_expiry", "keep_persistent", "delete_key"], "keep_persistent", 0),
    ],
    "postgresql": [
        ("A PostgreSQL 16 Serializable transaction ends with a serialization failure after several reads and writes. What unit of work should the application retry?",
         ["retry_last_statement", "retry_whole_transaction", "commit_partial_work"], "retry_whole_transaction", 0),
        ("In PostgreSQL 16 Read Committed, another transaction commits a row change between two ordinary SELECT statements in one transaction. Which snapshot should the second SELECT use?",
         ["use_first_statement_snapshot", "use_statement_snapshot", "read_uncommitted_data"], "use_statement_snapshot", 0),
    ],
    "sqs": [
        ("An Amazon SQS message has an explicit 30-second visibility timeout. Ten seconds after delivery it is undeleted, and no timeout adjustment occurs. Ignoring possible duplicate deliveries, how long remains until scheduled visibility returns?",
         ["wait_until_visible", "delete_automatically", "remain_invisible_forever"], "wait_until_visible", 20),
        ("An Amazon SQS standard-queue message is inside its visibility timeout. Does the documented delivery model still permit duplicate delivery? Choose the decision describing that possibility.",
         ["duplicate_delivery_possible", "exactly_once_guaranteed", "message_already_deleted"], "duplicate_delivery_possible", 0),
    ],
    "dict": [
        ("In Python 3.13, d is an ordinary empty dict. After d.setdefault('token', 7), what happens to the missing entry?",
         ["insert_default", "leave_dictionary_unchanged", "raise_key_error"], "insert_default", 0),
        ("In Python 3.13, d is an ordinary empty dict. After d.get('token', 7), what happens to the dictionary?",
         ["insert_default", "leave_dictionary_unchanged", "raise_key_error"], "leave_dictionary_unchanged", 0),
    ],
    "lru": [
        ("With Python 3.13 functools.lru_cache, two threads call the same uncached arguments before either execution has completed. What execution behavior is permitted?",
         ["may_execute_twice", "exactly_one_execution", "cache_must_corrupt"], "may_execute_twice", 0),
        ("A Python 3.13 functools.lru_cache entry still exists, with no eviction or cache_clear. Other references to its returned object are dropped. What does the cache do with its own reference?",
         ["retains_reference", "immediately_releases_reference", "only_keeps_weak_reference"], "retains_reference", 0),
    ],
    "asyncio": [
        ("In Python 3.13, asyncio.wait_for(task, 1) reaches its timeout, but task needs additional time to finish cancellation. What must wait_for do before it completes?",
         ["return_at_exact_timeout", "wait_for_cancellation", "silently_ignore_timeout"], "wait_for_cancellation", 0),
        ("In Python 3.13, asyncio.wait_for(asyncio.shield(task), 1) times out. The underlying task has no other cancellation or failure. What happens to that task?",
         ["underlying_task_cancelled", "underlying_task_continues", "task_becomes_synchronous"], "underlying_task_continues", 0),
    ],
    "http": [
        ("Under RFC 9110 section 9.2.2, a client sends PUT and the connection closes before any response is read. Is automatic retry of this idempotent request permitted?",
         ["automatic_retry_permitted", "automatic_retry_forbidden", "response_must_be_identical"], "automatic_retry_permitted", 0),
        ("Under RFC 9110 section 9.2.2, a POST fails before a response is read. The client has no evidence of idempotent operation semantics and cannot know whether the original operation was applied. What is the specified default retry decision?",
         ["automatic_retry_permitted", "avoid_automatic_retry", "retry_forever"], "avoid_automatic_retry", 0),
    ],
}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy", help="Optional explicit HTTP/HTTPS proxy for source snapshots")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output/name).exists() for name in ("corpus.jsonl", "questions.jsonl", "gold.jsonl")):
        parser.error("use a new output directory to preserve frozen data")
    raw = args.output/"source_snapshots"
    raw.mkdir(exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(
        {"http":args.proxy, "https":args.proxy}) if args.proxy else urllib.request.ProxyHandler())
    docs, questions, gold, sources = [], [], [], []
    for key, title, url, text in SOURCES:
        request = urllib.request.Request(url, headers={"User-Agent":"inference-scaling-document-audit/1.0"})
        with opener.open(request, timeout=60) as response:
            data = response.read()
            final_url = response.url
            content_type = response.headers.get("Content-Type")
        (raw/f"{key}.html").write_bytes(data)
        docid = "official-"+key
        docs.append({"id":docid, "title":title, "text":text, "source":url})
        sources.append({"id":docid, "url":url, "download_url":final_url, "content_type":content_type,
                        "snapshot":f"source_snapshots/{key}.html", "snapshot_sha256":hashlib.sha256(data).hexdigest(),
                        "retrieved_at":datetime.now(timezone.utc).isoformat(), "text_kind":"manual paraphrase"})
        for number, (scene, choices, correct, wait) in enumerate(CASES[key], 1):
            qid = f"official-{key}-{number}"
            question = scene+"\nReason carefully. End with ONLY a JSON object containing action and wait_seconds. Allowed actions: "+", ".join(choices)+". Set wait_seconds to the required remaining delay in seconds; use 0 when no mandatory timer is specified."
            questions.append({"id":qid, "dataset":"api_contract_diagnostic", "category":key, "question":question})
            gold.append({"id":qid, "answer":{"action":correct,"wait_seconds":wait}, "document_ids":[docid]})
    for name, rows in (("corpus",docs),("questions",questions),("gold",gold)):
        write_jsonl(args.output/f"{name}.jsonl", rows)
    metadata = {"dataset":"official-document-rules-v1", "count":len(questions), "documents":len(docs),
        "description":"Manually authored document-rule diagnostic, not a published benchmark; no service API calls", "sources":sources,
        "files":{p.relative_to(args.output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in args.output.rglob("*") if p.is_file()}}
    (args.output/"manifest.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"dataset":metadata["dataset"],"count":len(questions),"documents":len(docs)}))


if __name__ == "__main__":
    main()
