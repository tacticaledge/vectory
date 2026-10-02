"""Build and verify the GA image in AWS before staging it in Marketplace ECR.

This is a one-time seller-account build. It does not mutate ECS, a Marketplace
product, or an offer. The CodeBuild role is scoped to the Marketplace image
repository, its source archive, logs, and scan evidence.
"""

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vulnerability-maintenance"))
import maintain as m  # noqa: E402

SELLER_ACCOUNT = "063411696787"
MARKETPLACE_REGISTRY = "709825985650"
IMAGE = f"{MARKETPLACE_REGISTRY}.dkr.ecr.us-east-1.amazonaws.com/tactical-edge/vectory-self-hosted"
SOURCE_COMMIT = "c1691eeb04588d0c0643d3e236f9d64fca206dcc"
TAG = "ga-1.1.1-c1691ee"
EVIDENCE_BUCKET = "marketplace-images-tactical-edge"


def main():
    m.REPORTS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path("marketplace-source.json").read_text())
    if manifest != {"commit": SOURCE_COMMIT, "version": "1.1.1"}:
        raise RuntimeError("Source archive does not match reviewed GA release")
    if m.aws("sts", "get-caller-identity")["Account"] != SELLER_ACCOUNT:
        raise RuntimeError("Build is not running in the Marketplace seller account")
    if 'Development Status :: 5 - Production/Stable' not in Path("pyproject.toml").read_text():
        raise RuntimeError("Release is not marked GA")
    if '__version__ = "1.1.1"' not in Path("components/__init__.py").read_text():
        raise RuntimeError("Release version mismatch")

    build_id = os.environ["CODEBUILD_BUILD_ID"].split(":")[-1]
    image = f"{IMAGE}:{TAG}"
    password = m.command("aws", "ecr", "get-login-password", "--region", m.REGION, capture=True)
    m.command("docker", "login", "--username", "AWS", "--password-stdin", MARKETPLACE_REGISTRY + ".dkr.ecr.us-east-1.amazonaws.com", input=password)
    archive = Path("/tmp/trivy.tar.gz")
    urllib.request.urlretrieve(f"https://github.com/aquasecurity/trivy/releases/download/v{m.TRIVY_VERSION}/trivy_{m.TRIVY_VERSION}_Linux-64bit.tar.gz", archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != m.TRIVY_SHA256:
        raise RuntimeError("Trivy checksum mismatch")
    m.command("tar", "xzf", str(archive), "-C", "/tmp", "trivy")
    m.command("/tmp/trivy", "image", "--download-db-only")
    metadata = json.loads((Path.home() / ".cache/trivy/db/metadata.json").read_text())
    updated = dt.datetime.fromisoformat(metadata["UpdatedAt"].replace("Z", "+00:00"))
    if not dt.timedelta(0) <= dt.datetime.now(dt.timezone.utc) - updated <= dt.timedelta(hours=24):
        raise RuntimeError("Vulnerability database is stale")
    m.save("database.json", metadata)

    m.command("docker", "build", "--pull", "--no-cache", "--label",
              "org.opencontainers.image.revision=" + SOURCE_COMMIT, "-t", image, ".")
    testtools = Path("/tmp/vectory-testtools")
    m.command("python3", "-m", "pip", "install", "--no-cache-dir", "--target", str(testtools), "pip==26.2.1", "pytest==8.4.2")
    overlay = ("-v", f"{testtools}:/testtools:ro", "-e", "PYTHONPATH=/testtools")
    m.command("docker", "run", "--rm", *overlay, image, "-m", "pip", "check")
    m.command("docker", "run", "--rm", *overlay, image, "-m", "pytest", "-q")
    user = m.command("docker", "run", "--rm", image, "-c",
                     "import os,tempfile; tempfile.NamedTemporaryFile(dir='/app').close(); print(os.getuid())",
                     capture=True).strip()
    if user != "10001":
        raise RuntimeError("Container is not running as its expected unprivileged user")
    m.command("docker", "run", "-d", "--name", "vectory-marketplace-health",
              "-p", "127.0.0.1:18501:8501", image)
    try:
        for attempt in range(30):
            try:
                m.http_health("http://127.0.0.1:18501/_stcore/health")
                break
            except Exception:
                if attempt == 29:
                    raise
                time.sleep(2)
    finally:
        m.command("docker", "rm", "-f", "vectory-marketplace-health")

    candidate = m.gate(m.scan(image, "candidate-trivy.json"))
    m.save("candidate-summary.json", candidate)
    if not candidate["eligible"]:
        raise RuntimeError("Candidate image has HIGH, CRITICAL or UNKNOWN findings")
    m.command("docker", "push", image)
    details = m.aws("ecr", "batch-get-image", "--registry-id", MARKETPLACE_REGISTRY,
                    "--repository-name", "tactical-edge/vectory-self-hosted",
                    "--image-ids", f"imageTag={TAG}")
    if details.get("failures") or len(details.get("images", [])) != 1:
        raise RuntimeError("Marketplace registry image read-back failed")
    digest = details["images"][0]["imageId"]["imageDigest"]
    registry = m.gate(m.scan(f"{IMAGE}@{digest}", "registry-trivy.json"))
    m.save("registry-summary.json", registry)
    if not registry["eligible"]:
        raise RuntimeError("Marketplace registry image has HIGH, CRITICAL or UNKNOWN findings")
    result = {"status": "STAGED", "build": build_id, "sourceCommit": SOURCE_COMMIT,
              "image": f"{IMAGE}@{digest}", "candidate": candidate["counts"],
              "registry": registry["counts"]}
    m.save("status.json", result)
    m.command("aws", "s3", "cp", str(m.REPORTS),
              f"s3://{EVIDENCE_BUCKET}/vectory/build-evidence/{build_id}/", "--recursive",
              "--region", m.REGION)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
