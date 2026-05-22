import os
from contextlib import asynccontextmanager
from pathlib import Path

import msgspec
from loguru import logger
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from red_hela.adapters.http.responses import RESPONSE_BODIES, RESPONSE_FALLBACK, RESPONSE_READY
from red_hela.adapters.persistence.artifact_loader import ArtifactLoader
from red_hela.domain.transaction import FraudScoreRequest
from red_hela.domain.vector_search import VectorSearch
from red_hela.domain.vectorize import TransactionVectorizer
from red_hela.infrastructure.logging_setup import configure_logging

decoder = msgspec.json.Decoder(type=FraudScoreRequest)


def _project_root() -> Path:
    configured = os.environ.get("RED_HELA_ROOT")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[4]


@asynccontextmanager
async def lifespan(application: Starlette):
    configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    root = _project_root()
    resources = root / "resources"
    logger.info("booting red-hela from {}", root)
    loader = ArtifactLoader(resources)
    search = VectorSearch(loader)
    search.warmup()
    application.state.ready = True
    application.state.search = search
    application.state.vectorizer = TransactionVectorizer()
    logger.info("ready to serve")
    yield


def ready(request: Request) -> Response:
    if not getattr(request.app.state, "ready", False):
        return Response(status_code=503)
    return Response(content=RESPONSE_READY, media_type="application/json")


async def fraud_score(request: Request) -> Response:
    if not getattr(request.app.state, "ready", False):
        return Response(status_code=503)
    try:
        payload = decoder.decode(await request.body())
        vector = request.app.state.vectorizer.vectorize(payload)
        fraud_score_value, approved = request.app.state.search.score(vector, k=5)
        body = RESPONSE_BODIES.get((approved, fraud_score_value))
        if body is None:
            body = msgspec.json.encode(
                {"approved": approved, "fraud_score": fraud_score_value}
            )
        return Response(content=body, media_type="application/json")
    except Exception as exc:
        logger.opt(exception=exc).warning("fraud-score fallback")
        return Response(content=RESPONSE_FALLBACK, media_type="application/json")


app = Starlette(
    lifespan=lifespan,
    routes=[
        Route("/ready", ready, methods=["GET"]),
        Route("/fraud-score", fraud_score, methods=["POST"]),
    ],
)
