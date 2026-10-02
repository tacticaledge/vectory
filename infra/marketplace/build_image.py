"""Build and verify the GA image in AWS before staging it in Marketplace ECR.

This is a one-time seller-account build. It does not mutate ECS, a Marketplace
product, or an offer. The CodeBuild role is scoped to the Marketplace image
repository, its source archive, logs, and scan evidence.
"""

import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vulnerability-maintenance"))
import maintain as m  # noqa: E402

SELLER_ACCOUNT = "063411696787"
MARKETPLACE_REGISTRY = "709825985650"
IMAGE = f"{MARKETPLACE_REGISTRY}.dkr.ecr.us-east-1.amazonaws.com/tactical-edge/vectory-self-hosted"
SOURCE_COMMIT = "c1691eeb04588d0c0643d3e236f9d64fca206dcc"
TAG = "ga-1.1.1-c1691ee"
EVIDENCE_BUCKET = "marketplace-images-tactical-edge"


class BuildConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")
    codebuild_build_id: str = Field(validation_alias="CODEBUILD_BUILD_ID")
    vectory_expected_source_tree_sha256: str = Field(validation_alias="VECTORY_EXPECTED_SOURCE_TREE_SHA256")

    @property
    def build_id(self):
        project, separator, build_id = self.codebuild_build_id.partition(":")
        if project != "vectory-marketplace-ga-build" or not separator or not re.fullmatch(r"[0-9a-f-]{36}", build_id):
            raise RuntimeError("Invalid CodeBuild build identity")
        return build_id


def verify_source(config):
    manifest = json.loads(Path("marketplace-source.json").read_text())
    if manifest.get("commit") != SOURCE_COMMIT or manifest.get("version") != "1.1.1":
        raise RuntimeError("Source archive does not match reviewed GA release")
    expected = manifest.get("files")
    if not isinstance(expected, list) or not expected or expected != sorted(set(expected)):
        raise RuntimeError("Invalid source file inventory")
    actual = sorted(p.as_posix() for p in Path(".").rglob("*") if p.is_file()
                    and "__pycache__" not in p.parts and p.as_posix() != "marketplace-source.json")
    if actual != expected:
        raise RuntimeError("Source archive file inventory changed")
    source_hash = hashlib.sha256()
    for name in expected:
        source_hash.update(name.encode() + b"\0" + Path(name).read_bytes() + b"\0")
    if not re.fullmatch(r"[a-f0-9]{64}", config.vectory_expected_source_tree_sha256):
        raise RuntimeError("Invalid approved source tree hash")
    if source_hash.hexdigest() != config.vectory_expected_source_tree_sha256:
        raise RuntimeError("Source archive differs from independently approved tree hash")
    return source_hash.hexdigest()


def image_by_tag(tag):
    response = m.aws("ecr", "batch-get-image", "--registry-id", MARKETPLACE_REGISTRY,
                     "--repository-name", "tactical-edge/vectory-self-hosted",
                     "--image-ids", f"imageTag={tag}")
    if response.get("failures") or len(response.get("images", [])) != 1:
        raise RuntimeError("Marketplace registry image read-back failed")
    return response["images"][0]


def verify_and_stage(config):
    source_hash = verify_source(config)
    if m.aws("sts", "get-caller-identity")["Account"] != SELLER_ACCOUNT:
        raise RuntimeError("Build is not running in the Marketplace seller account")
    if 'Development Status :: 5 - Production/Stable' not in Path("pyproject.toml").read_text():
        raise RuntimeError("Release is not marked GA")
    if '__version__ = "1.1.1"' not in Path("components/__init__.py").read_text():
        raise RuntimeError("Release version mismatch")

    candidate_tag = f"candidate-{config.build_id}"
    image = f"{IMAGE}:{candidate_tag}"
    existing = m.aws("ecr", "batch-get-image", "--registry-id", MARKETPLACE_REGISTRY,
                     "--repository-name", "tactical-edge/vectory-self-hosted",
                     "--image-ids", f"imageTag={TAG}")
    if existing.get("images") or len(existing.get("failures", [])) != 1 or existing["failures"][0].get("failureCode") != "ImageNotFound":
        raise RuntimeError("GA tag exists or its absence could not be proven")
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
    # Marketplace-owned ECR permits sellers to push but not delete images.
    # A failed registry scan leaves only this clearly marked candidate tag;
    # no buyer delivery version references it.
    m.command("docker", "push", image)
    registered = image_by_tag(candidate_tag)
    digest = registered["imageId"]["imageDigest"]
    registry = m.gate(m.scan(f"{IMAGE}@{digest}", "registry-trivy.json"))
    m.save("registry-summary.json", registry)
    if not registry["eligible"]:
        raise RuntimeError("Marketplace registry image has HIGH, CRITICAL or UNKNOWN findings")
    m.aws("ecr", "put-image", "--registry-id", MARKETPLACE_REGISTRY,
          "--repository-name", "tactical-edge/vectory-self-hosted",
          "--image-manifest", registered["imageManifest"], "--image-tag", TAG)
    if image_by_tag(TAG)["imageId"]["imageDigest"] != digest:
        raise RuntimeError("GA tag digest does not match scanned candidate")
    return {"status": "STAGED", "build": config.build_id, "sourceCommit": SOURCE_COMMIT,
            "sourceTreeSha256": source_hash,
            "image": f"{IMAGE}@{digest}", "candidate": candidate["counts"],
            "registry": registry["counts"]}


def main():
    config = BuildConfig()
    m.REPORTS.mkdir(parents=True, exist_ok=True)
    failed = False
    try:
        result = verify_and_stage(config)
    except Exception as exc:
        failed = True
        result = {"status": "FAILED", "build": config.build_id, "error": str(exc)}
        raise
    finally:
        m.save("status.json", result)
        print(json.dumps(result), flush=True)
        try:
            m.command("aws", "s3", "cp", str(m.REPORTS),
                      f"s3://{EVIDENCE_BUCKET}/vectory/build-evidence/{config.build_id}/",
                      "--recursive", "--region", m.REGION)
        except Exception as exc:
            print(json.dumps({"evidenceUploadError": str(exc), "build": config.build_id}), flush=True)
            if not failed:
                raise


if __name__ == "__main__":
    main()
