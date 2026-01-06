import json
import os
from google import genai
from google.cloud import aiplatform_v1
from google.oauth2 import service_account
import google.auth
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

# ===== CONFIG (MATCH YOUR DEPLOYMENT) =====
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID", "1010849458554")
REGION = os.getenv("REGION", "us-central1")
INDEX_ID = os.getenv("INDEX_ID", "6537302513593352192")   # THIS IS THE INDEX ID (not endpoint)

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "16"))
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "200"))

VERTEX_SA_PATH = os.getenv("VERTEX_SA_PATH", "vertexmanager-key.json")
DEBUG_AUTH = os.getenv("DEBUG_AUTH", "0").strip().lower() in {"1", "true", "yes"}

# INGEST_MODE:
# - "batch"  : write JSONL file for Vertex Vector Search batch import (works with Update method: Batch)
# - "stream" : call upsert_datapoints (requires StreamUpdate enabled on the index)
INGEST_MODE = os.getenv("INGEST_MODE", "batch").strip().lower()
EXPORT_JSONL_PATH = os.getenv("EXPORT_JSONL_PATH", "vertex_datapoints_batch.json")

# ===== LOAD CORPUS =====
with open("gemini_cleaned_text.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

ids = list(corpus.keys())
texts = list(corpus.values())

print(f"Loaded {len(texts)} documents")

# ===== TF-IDF (MUST MATCH server.py) =====
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)
tfidf = vectorizer.fit_transform(texts)

print("TF-IDF fitted")

# ===== GEMINI EMBEDDINGS =====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "Missing GEMINI_API_KEY. Set it in your shell or in a .env file (this script loads .env)."
    )
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# ===== VERTEX CLIENT =====
scopes = ["https://www.googleapis.com/auth/cloud-platform"]

if VERTEX_SA_PATH and os.path.exists(VERTEX_SA_PATH):
    creds = service_account.Credentials.from_service_account_file(
        VERTEX_SA_PATH,
        scopes=scopes,
    )
else:
    # ADC (recommended on GCP / Cloud Run / when using gcloud locally)
    creds, adc_project = google.auth.default(scopes=scopes)

if DEBUG_AUTH:
    sa_email = getattr(creds, "service_account_email", None)
    print("Auth: using project:", PROJECT_ID)
    print("Auth: using service account:", sa_email or "(not a service account; likely user ADC)")

index_client = aiplatform_v1.IndexServiceClient(
    credentials=creds,
    client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"},
)

# ===== BUILD DATAPOINTS =====
index_resource = f"projects/{PROJECT_ID}/locations/{REGION}/indexes/{INDEX_ID}"


def _extract_embedding_values(embed_response, idx: int):
    # google.genai can return embeddings via `.embeddings` (list) or `.embedding` (single)
    if hasattr(embed_response, "embeddings") and embed_response.embeddings:
        return embed_response.embeddings[idx].values
    if hasattr(embed_response, "embedding"):
        return embed_response.embedding.values
    raise RuntimeError("Unexpected embed response shape; could not find embeddings values")


total = len(texts)
upserted = 0

def _to_vertex_json_dict(*, datapoint_id: str, feature_vector, sparse_values, sparse_dimensions):
    # Vertex Vector Search batch import expects newline-delimited JSON objects.
    # See: https://docs.cloud.google.com/vertex-ai/docs/vector-search/format-structure
    return {
        "id": str(datapoint_id),
        "embedding": feature_vector,
        "sparse_embedding": {
            "values": sparse_values,
            "dimensions": sparse_dimensions,
        },
    }


_jsonl_fh = None
if INGEST_MODE == "batch":
    _jsonl_fh = open(EXPORT_JSONL_PATH, "w", encoding="utf-8")
    print(f"Batch mode: exporting datapoints to {EXPORT_JSONL_PATH}")
elif INGEST_MODE == "stream":
    print("Stream mode: upserting datapoints via upsert_datapoints (StreamUpdate must be enabled)")
else:
    raise RuntimeError(f"Invalid INGEST_MODE={INGEST_MODE}. Use 'batch' or 'stream'.")

for start in range(0, total, EMBED_BATCH_SIZE):
    end = min(start + EMBED_BATCH_SIZE, total)
    text_batch = texts[start:end]

    embed_response = genai_client.models.embed_content(
        model="models/embedding-001",
        contents=text_batch,
        config={
            "task_type": "RETRIEVAL_DOCUMENT",
            "output_dimensionality": 768,
        },
    )

    batch_datapoints = []
    for local_i in range(end - start):
        global_i = start + local_i
        feature_vector = _extract_embedding_values(embed_response, local_i)
        sparse_values = tfidf[global_i].data.tolist()
        sparse_dimensions = tfidf[global_i].indices.tolist()

        if INGEST_MODE == "batch":
            rec = _to_vertex_json_dict(
                datapoint_id=ids[global_i],
                feature_vector=feature_vector,
                sparse_values=sparse_values,
                sparse_dimensions=sparse_dimensions,
            )
            _jsonl_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            upserted += 1
        else:
            batch_datapoints.append(
                aiplatform_v1.IndexDatapoint(
                    datapoint_id=ids[global_i],
                    feature_vector=feature_vector,
                    sparse_embedding=aiplatform_v1.IndexDatapoint.SparseEmbedding(
                        values=sparse_values,
                        dimensions=sparse_dimensions,
                    ),
                )
            )

    if INGEST_MODE == "batch":
        if upserted % 200 == 0 or upserted == total:
            print(f"Exported {upserted}/{total} datapoints")
    else:
        for up_start in range(0, len(batch_datapoints), UPSERT_BATCH_SIZE):
            up_end = min(up_start + UPSERT_BATCH_SIZE, len(batch_datapoints))
            req = aiplatform_v1.UpsertDatapointsRequest(
                index=index_resource,
                datapoints=batch_datapoints[up_start:up_end],
            )
            index_client.upsert_datapoints(request=req)
            upserted += (up_end - up_start)
            print(f"Upserted {upserted}/{total} datapoints")

if _jsonl_fh:
    _jsonl_fh.close()
    print(f"Wrote {upserted} JSONL records to {EXPORT_JSONL_PATH}")

print("✅ DATA INGESTION COMPLETE")
