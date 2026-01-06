import os
from dotenv import load_dotenv
import google.auth
from google.oauth2 import service_account
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index import MatchingEngineIndex


def _bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    load_dotenv()

    project_id = os.getenv("PROJECT_ID")
    region = os.getenv("REGION", "us-central1")
    index_id = os.getenv("INDEX_ID")

    if not project_id:
        raise SystemExit("Missing PROJECT_ID")
    if not index_id:
        raise SystemExit("Missing INDEX_ID")

    contents_delta_uri = os.getenv("CONTENTS_DELTA_URI")
    if not contents_delta_uri:
        raise SystemExit(
            "Missing CONTENTS_DELTA_URI (should be a GCS *directory*, e.g. gs://<bucket>/vertex/batch_001/)"
        )

    is_complete_overwrite = _bool_env("IS_COMPLETE_OVERWRITE", "1")
    debug_auth = _bool_env("DEBUG_AUTH", "0")

    vertex_sa_path = os.getenv("VERTEX_SA_PATH", "vertexmanager-key.json")
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]

    if vertex_sa_path and os.path.exists(vertex_sa_path):
        credentials = service_account.Credentials.from_service_account_file(vertex_sa_path, scopes=scopes)
    else:
        credentials, _ = google.auth.default(scopes=scopes)

    if debug_auth:
        sa_email = getattr(credentials, "service_account_email", None)
        print("Auth: project:", project_id)
        print("Auth: principal:", sa_email or "(not a service account; likely user ADC)")

    aiplatform.init(project=project_id, location=region, credentials=credentials)

    index_resource_name = f"projects/{project_id}/locations/{region}/indexes/{index_id}"
    index = MatchingEngineIndex(index_name=index_resource_name, credentials=credentials)

    print("Triggering batch update:")
    print("- Index:", index_resource_name)
    print("- contentsDeltaUri:", contents_delta_uri)
    print("- isCompleteOverwrite:", is_complete_overwrite)

    op = index.update_embeddings(
        contents_delta_uri=contents_delta_uri,
        is_complete_overwrite=is_complete_overwrite,
    )

    print("Update operation started:")
    print(op)
    print("Note: index rebuild can take minutes to hours depending on size.")


if __name__ == "__main__":
    main()
