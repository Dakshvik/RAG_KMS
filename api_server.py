# # api_server.py
# import os
# import logging
# from fastapi import FastAPI, HTTPException, Header, Request
# from fastapi.responses import JSONResponse
# from dotenv import load_dotenv
# from threading import Lock

# load_dotenv()

# API_TOKEN = os.getenv("API_TOKEN", "1234567890")

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("api_server")

# app = FastAPI()  # No lifespan – pipeline loaded on first use

# # Lazy pipeline holder
# pipeline = None
# pipeline_lock = Lock()

# def get_pipeline():
#     global pipeline
#     if pipeline is None:
#         with pipeline_lock:
#             if pipeline is None:
#                 logger.info("First request – loading RAG pipeline (this may take a minute)...")
#                 # Import your pipeline class
#                 from rag_pipeline_v7 import HybridRAGPipeline
#                 pipeline = HybridRAGPipeline()
#                 logger.info("Pipeline ready.")
#     return pipeline

# # Token verification
# def verify_token(authorization: str = Header(None)):
#     if not authorization or not authorization.startswith("Bearer "):
#         raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
#     token = authorization.split("Bearer ")[-1]
#     if token != API_TOKEN:
#         raise HTTPException(status_code=403, detail="Invalid token")

# # Health check (no token needed)
# @app.get("/health")
# async def health():
#     return {"status": "healthy"}

# # Main query endpoint
# @app.post("/query")
# async def query(request: Request, authorization: str = Header(None)):
#     verify_token(authorization)
#     body = await request.json()
#     query_text = body.get("query")
#     if not query_text:
#         raise HTTPException(status_code=400, detail="Missing 'query' in request body")

#     pl = get_pipeline()
#     result = pl.process_query(query_text)
#     return JSONResponse(content=result)


# api_server.py

# import os
# import time
# import logging
# import traceback
# import asyncio

# from threading import Lock

# from dotenv import load_dotenv

# from fastapi import (
#     FastAPI,
#     HTTPException,
#     Header,
# )

# from fastapi.responses import JSONResponse

# from fastapi.concurrency import run_in_threadpool

# from pydantic import BaseModel

# # =========================================================
# # LOAD ENV
# # =========================================================
# load_dotenv()

# API_TOKEN = os.getenv("API_TOKEN")

# if not API_TOKEN:
#     raise RuntimeError(
#         "API_TOKEN missing in .env"
#     )

# # =========================================================
# # LOGGING
# # =========================================================
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler(
#             "api_server.log",
#             encoding="utf-8"
#         )
#     ]
# )

# logger = logging.getLogger("api_server")

# logger.info("=" * 80)
# logger.info("API SERVER STARTING")
# logger.info("=" * 80)

# # =========================================================
# # FASTAPI
# # =========================================================
# app = FastAPI()

# # =========================================================
# # GLOBALS
# # =========================================================
# pipeline = None

# pipeline_lock = Lock()

# # Prevent GPU overload
# inference_lock = Lock()

# # =========================================================
# # REQUEST MODEL
# # =========================================================
# class QueryRequest(BaseModel):
#     query: str

# # =========================================================
# # LOAD PIPELINE LAZILY
# # =========================================================
# def get_pipeline():

#     global pipeline

#     if pipeline is None:

#         with pipeline_lock:

#             if pipeline is None:

#                 logger.info("=" * 80)
#                 logger.info(
#                     "FIRST REQUEST -> LOADING PIPELINE"
#                 )
#                 logger.info("=" * 80)

#                 start = time.perf_counter()

#                 try:

#                     from rag_pipeline_v8 import (
#                         HybridRAGPipeline
#                     )

#                     pipeline = HybridRAGPipeline()

#                     logger.info(
#                         f"PIPELINE READY "
#                         f"({round(time.perf_counter()-start,2)}s)"
#                     )

#                 except Exception:

#                     logger.error(
#                         "PIPELINE LOAD FAILED"
#                     )

#                     traceback.print_exc()

#                     raise

#     return pipeline

# # =========================================================
# # TOKEN VERIFICATION
# # =========================================================
# def verify_token(
#     authorization: str = Header(None)
# ):

#     if not authorization:

#         raise HTTPException(
#             status_code=401,
#             detail="Missing authorization header"
#         )

#     if not authorization.startswith("Bearer "):

#         raise HTTPException(
#             status_code=401,
#             detail="Invalid authorization format"
#         )

#     token = authorization.split("Bearer ")[-1]

#     if token != API_TOKEN:

#         raise HTTPException(
#             status_code=403,
#             detail="Invalid token"
#         )

# # =========================================================
# # GLOBAL EXCEPTION HANDLER
# # =========================================================
# @app.exception_handler(Exception)
# async def global_exception_handler(
#     request,
#     exc
# ):

#     logger.error("=" * 80)
#     logger.error("UNHANDLED EXCEPTION")
#     logger.error("=" * 80)

#     traceback.print_exc()

#     return JSONResponse(
#         status_code=500,
#         content={
#             "success": False,
#             "error": "Internal server error"
#         }
#     )

# # =========================================================
# # HEALTH CHECK
# # =========================================================
# @app.get("/health")
# async def health():

#     try:

#         import torch

#         gpu_available = torch.cuda.is_available()

#         gpu_name = None

#         if gpu_available:
#             gpu_name = torch.cuda.get_device_name(0)

#         return {
#             "status": "healthy",
#             "pipeline_loaded": pipeline is not None,
#             "gpu_available": gpu_available,
#             "gpu_name": gpu_name,
#         }

#     except Exception:

#         traceback.print_exc()

#         return {
#             "status": "error"
#         }

# # =========================================================
# # WARMUP
# # =========================================================
# @app.post("/warmup")
# async def warmup():

#     logger.info("WARMUP REQUEST")

#     get_pipeline()

#     return {
#         "success": True,
#         "message": "Pipeline warmed up"
#     }

# # =========================================================
# # MAIN QUERY ENDPOINT
# # =========================================================
# @app.post("/query")
# async def query(
#     req: QueryRequest,
#     authorization: str = Header(None)
# ):

#     # -----------------------------------------------------
#     # AUTH
#     # -----------------------------------------------------
#     verify_token(authorization)

#     # -----------------------------------------------------
#     # VALIDATION
#     # -----------------------------------------------------
#     query_text = req.query.strip()

#     if not query_text:

#         raise HTTPException(
#             status_code=400,
#             detail="Query is empty"
#         )

#     if len(query_text) > 2000:

#         raise HTTPException(
#             status_code=400,
#             detail="Query too long"
#         )

#     logger.info("=" * 80)
#     logger.info(f"INCOMING QUERY: {query_text}")
#     logger.info("=" * 80)

#     start_total = time.perf_counter()

#     try:

#         # -------------------------------------------------
#         # LOAD PIPELINE
#         # -------------------------------------------------
#         pl = get_pipeline()

#         # -------------------------------------------------
#         # SERIALIZE GPU ACCESS
#         # -------------------------------------------------
#         with inference_lock:

#             logger.info(
#                 "STARTING PIPELINE INFERENCE"
#             )

#             # ---------------------------------------------
#             # TIMEOUT PROTECTION
#             # ---------------------------------------------
#             result = await asyncio.wait_for(

#                 run_in_threadpool(
#                     pl.process_query,
#                     query_text
#                 ),

#                 timeout=180
#             )

#         total_time = round(
#             time.perf_counter() - start_total,
#             2
#         )

#         logger.info(
#             f"QUERY COMPLETED "
#             f"({total_time}s)"
#         )

#         return JSONResponse(
#             content={
#                 "success": True,
#                 "query": query_text,
#                 "processing_time": total_time,
#                 "result": result
#             }
#         )

#     except asyncio.TimeoutError:

#         logger.error("QUERY TIMEOUT")

#         return JSONResponse(
#             status_code=504,
#             content={
#                 "success": False,
#                 "error": "Request timed out"
#             }
#         )

#     except Exception:

#         logger.error("QUERY FAILED")

#         traceback.print_exc()

#         return JSONResponse(
#             status_code=500,
#             content={
#                 "success": False,
#                 "error": "Query processing failed"
#             }
#         )

# # =========================================================
# # ROOT
# # =========================================================
# @app.get("/")
# async def root():

#     return {
#         "message": "Hybrid RAG API running"
#     }

# # =========================================================
# # MAIN
# # =========================================================
# if __name__ == "__main__":

#     import uvicorn

#     logger.info("=" * 80)
#     logger.info("STARTING UVICORN")
#     logger.info("=" * 80)

#     uvicorn.run(
#         "api_server:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=False,
#         workers=1,
#     )


import os
import time
import logging
import traceback
import asyncio

from threading import Lock

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    HTTPException,
    Header,
)

from fastapi.responses import JSONResponse

from fastapi.concurrency import run_in_threadpool

from pydantic import BaseModel

# =========================================================
# LOAD ENV
# =========================================================
load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")

if not API_TOKEN:
    raise RuntimeError(
        "API_TOKEN missing in .env"
    )

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            "api_server.log",
            encoding="utf-8"
        )
    ]
)

logger = logging.getLogger("api_server")

logger.info("=" * 80)
logger.info("API SERVER STARTING")
logger.info("=" * 80)

# =========================================================
# FASTAPI
# =========================================================
app = FastAPI()

# =========================================================
# GLOBALS
# =========================================================
pipeline = None

pipeline_lock = Lock()

# IMPORTANT FOR 8GB VRAM
inference_lock = Lock()

# =========================================================
# REQUEST MODEL
# =========================================================
class QueryRequest(BaseModel):
    query: str

# =========================================================
# LOAD PIPELINE
# =========================================================
def get_pipeline():

    global pipeline

    if pipeline is None:

        with pipeline_lock:

            if pipeline is None:

                logger.info("=" * 80)
                logger.info(
                    "LOADING HYBRID RAG PIPELINE"
                )
                logger.info("=" * 80)

                start = time.perf_counter()

                try:

                    from rag_pipeline_v8 import (
                        HybridRAGPipeline
                    )

                    pipeline = HybridRAGPipeline()

                    logger.info(
                        f"PIPELINE READY "
                        f"({round(time.perf_counter()-start,2)}s)"
                    )

                except Exception:

                    logger.error(
                        "PIPELINE LOAD FAILED"
                    )

                    traceback.print_exc()

                    raise

    return pipeline

# =========================================================
# TOKEN VERIFY
# =========================================================
def verify_token(
    authorization: str = Header(None)
):

    if not authorization:

        raise HTTPException(
            status_code=401,
            detail="Missing authorization header"
        )

    if not authorization.startswith("Bearer "):

        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format"
        )

    token = authorization.split("Bearer ")[-1]

    if token != API_TOKEN:

        raise HTTPException(
            status_code=403,
            detail="Invalid token"
        )

# =========================================================
# EXCEPTION HANDLER
# =========================================================
@app.exception_handler(Exception)
async def global_exception_handler(
    request,
    exc
):

    logger.error("=" * 80)
    logger.error("UNHANDLED EXCEPTION")
    logger.error("=" * 80)

    traceback.print_exc()

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error"
        }
    )

# =========================================================
# ROOT
# =========================================================
@app.get("/")
async def root():

    return {
        "message":
        "Hybrid RAG API running"
    }

# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
async def health():

    try:

        import torch

        gpu_available = (
            torch.cuda.is_available()
        )

        gpu_name = None

        if gpu_available:

            gpu_name = (
                torch.cuda.get_device_name(0)
            )

        return {

            "status": "healthy",

            "pipeline_loaded":
            pipeline is not None,

            "gpu_available":
            gpu_available,

            "gpu_name":
            gpu_name,
        }

    except Exception:

        traceback.print_exc()

        return {
            "status": "error"
        }

# =========================================================
# WARMUP
# =========================================================
@app.post("/warmup")
async def warmup():

    logger.info("WARMUP REQUEST")

    get_pipeline()

    return {

        "success": True,

        "message":
        "Pipeline warmed up"
    }

# =========================================================
# MAIN QUERY
# =========================================================
@app.post("/query")
async def query(
    req: QueryRequest,
    authorization: str = Header(None)
):

    # -----------------------------------------------------
    # AUTH
    # -----------------------------------------------------
    verify_token(authorization)

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------
    query_text = req.query.strip()

    if not query_text:

        raise HTTPException(
            status_code=400,
            detail="Query is empty"
        )

    if len(query_text) > 3000:

        raise HTTPException(
            status_code=400,
            detail="Query too long"
        )

    logger.info("=" * 80)
    logger.info(f"QUERY: {query_text}")
    logger.info("=" * 80)

    start_total = time.perf_counter()

    try:

        # -------------------------------------------------
        # PIPELINE
        # -------------------------------------------------
        pl = get_pipeline()

        # -------------------------------------------------
        # IMPORTANT:
        # SERIALIZE GPU ACCESS
        # -------------------------------------------------
        with inference_lock:

            logger.info(
                "STARTING PIPELINE"
            )

            result = await asyncio.wait_for(

                run_in_threadpool(
                    pl.process_query,
                    query_text
                ),

                timeout=300
            )

        total_time = round(
            time.perf_counter()
            - start_total,
            2
        )

        logger.info(
            f"QUERY COMPLETED "
            f"({total_time}s)"
        )

        return JSONResponse(
            content={

                "success": True,

                "query": query_text,

                "processing_time":
                total_time,

                "result":
                result
            }
        )

    except asyncio.TimeoutError:

        logger.error(
            "QUERY TIMEOUT"
        )

        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error":
                "Request timed out"
            }
        )

    except Exception:

        logger.error(
            "QUERY FAILED"
        )

        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error":
                "Query processing failed"
            }
        )

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    import uvicorn

    logger.info("=" * 80)
    logger.info("STARTING UVICORN")
    logger.info("=" * 80)

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
    )