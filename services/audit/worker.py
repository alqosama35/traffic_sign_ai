import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pika

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
LOG_FILE     = Path(os.environ.get("AUDIT_LOG_FILE", "/app/logs/predictions.jsonl"))
EXCHANGE     = "traffic_sign"


def _connect_with_retry(max_attempts: int = 10, base_delay: int = 3):
    for attempt in range(max_attempts):
        try:
            params = pika.URLParameters(RABBITMQ_URL)
            params.heartbeat = 60
            params.blocked_connection_timeout = 10
            return pika.BlockingConnection(params)
        except Exception as exc:
            wait = base_delay * (2 ** min(attempt, 4))
            print(f"[audit] RabbitMQ not ready ({exc}), retry in {wait}s ...", flush=True)
            time.sleep(wait)
    raise RuntimeError("Could not connect to RabbitMQ after retries")


def callback(ch, method, properties, body):
    try:
        event = json.loads(body)
        event["logged_at"] = datetime.now(timezone.utc).isoformat()
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        cls  = event.get("predicted_class", "?")
        conf = event.get("confidence", 0)
        print(f"[audit] logged: {cls} ({conf:.1%})", flush=True)
    except Exception as exc:
        print(f"[audit] error processing message: {exc}", file=sys.stderr, flush=True)
        # nack without requeue → goes to DLQ, never silently lost
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main():
    conn = _connect_with_retry()
    ch   = conn.channel()

    # Declare topology — idempotent, safe to call on every start
    ch.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)
    ch.exchange_declare(exchange="traffic_sign_dlx", exchange_type="direct", durable=True)
    ch.queue_declare(
        queue="prediction.audit",
        durable=True,
        arguments={
            "x-dead-letter-exchange":     "traffic_sign_dlx",
            "x-dead-letter-routing-key":  "prediction.dead",
        },
    )
    ch.queue_bind(queue="prediction.audit", exchange=EXCHANGE,
                  routing_key="prediction.completed")
    ch.queue_declare(queue="prediction.dead", durable=True)
    ch.queue_bind(queue="prediction.dead", exchange="traffic_sign_dlx",
                  routing_key="prediction.dead")

    ch.basic_qos(prefetch_count=10)  # audit writes are cheap — batch them
    ch.basic_consume(queue="prediction.audit", on_message_callback=callback)
    print("[audit] waiting for prediction events ...", flush=True)
    ch.start_consuming()


if __name__ == "__main__":
    main()
