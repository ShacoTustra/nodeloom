"""Where files live: a local directory (one machine) or S3-compatible object storage shared by all processes of a
cluster (MinIO, Garage, R2, AWS: any S3 works).

Keys are "<user id>/<kind>/<file>" (files.py). The S3 client is boto3 run in threads: the traffic is small.
"""

from __future__ import annotations

import asyncio
import mimetypes
import re
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse, Response, StreamingResponse

from nodeloom.config import get_settings

CHUNK = 256 * 1024


class MissingFile(Exception):
    pass


def content_type(key: str) -> str:
    return mimetypes.guess_type(key)[0] or "application/octet-stream"


class LocalStore:
    def __init__(self, root: str):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise MissingFile(key)
        return p

    async def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    async def put_file(self, key: str, path: Path) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), p)

    async def get(self, key: str) -> bytes:
        p = self._path(key)
        if not p.is_file():
            raise MissingFile(key)
        return p.read_bytes()

    async def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    async def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    async def delete_prefix(self, prefix: str) -> None:
        shutil.rmtree(self._path(prefix.rstrip("/")), ignore_errors=True)

    async def response(self, key: str, request: Request, filename: str | None) -> Response:
        p = self._path(key)
        if not p.is_file():
            raise MissingFile(key)
        return FileResponse(p, media_type=content_type(key), filename=filename,
                            headers={"Cache-Control": "private, max-age=31536000, immutable"})


class S3Store:
    def __init__(self, endpoint: str, bucket: str, access_key: str, secret_key: str, region: str):
        import boto3
        from botocore.config import Config

        self.bucket = bucket
        self.s3 = boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=access_key,
                               aws_secret_access_key=secret_key, region_name=region,
                               config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 3}))

    async def put(self, key: str, data: bytes) -> None:
        await asyncio.to_thread(self.s3.put_object, Bucket=self.bucket, Key=key, Body=data,
                                ContentType=content_type(key))

    async def put_file(self, key: str, path: Path) -> None:
        try:
            await asyncio.to_thread(self.s3.upload_file, str(path), self.bucket, key,
                                    ExtraArgs={"ContentType": content_type(key)})
        finally:
            path.unlink(missing_ok=True)

    async def get(self, key: str) -> bytes:
        def read():
            try:
                return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            except self.s3.exceptions.NoSuchKey:
                raise MissingFile(key) from None
        return await asyncio.to_thread(read)

    async def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            await asyncio.to_thread(self.s3.head_object, Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self.s3.delete_object, Bucket=self.bucket, Key=key)

    async def delete_prefix(self, prefix: str) -> None:
        def run():
            token = None
            while True:
                kw = {"Bucket": self.bucket, "Prefix": prefix}
                if token:
                    kw["ContinuationToken"] = token
                page = self.s3.list_objects_v2(**kw)
                keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
                if keys:
                    self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": keys, "Quiet": True})
                if not page.get("IsTruncated"):
                    return
                token = page.get("NextContinuationToken")
        await asyncio.to_thread(run)

    async def response(self, key: str, request: Request, filename: str | None) -> Response:
        from botocore.exceptions import ClientError
        rng = request.headers.get("range")
        kw = {"Bucket": self.bucket, "Key": key}
        if rng and re.fullmatch(r"bytes=\d*-\d*", rng.strip()):
            kw["Range"] = rng.strip()
        try:
            obj = await asyncio.to_thread(self.s3.get_object, **kw)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code == "InvalidRange":
                return Response(status_code=416)
            raise MissingFile(key) from None
        headers = {"Accept-Ranges": "bytes", "Content-Length": str(obj["ContentLength"]),
                   "Cache-Control": "private, max-age=31536000, immutable"}
        if obj.get("ContentRange"):
            headers["Content-Range"] = obj["ContentRange"]
        if filename:
            headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        body = obj["Body"]

        async def stream():
            try:
                while chunk := await asyncio.to_thread(body.read, CHUNK):
                    yield chunk
            finally:
                body.close()

        return StreamingResponse(stream(), status_code=206 if obj.get("ContentRange") else 200,
                                 media_type=obj.get("ContentType") or content_type(key), headers=headers)


@lru_cache
def _store(root: str, endpoint: str | None, bucket: str, ak: str | None, sk: str | None, region: str):
    if endpoint:
        return S3Store(endpoint, bucket, ak or "", sk or "", region)
    return LocalStore(root)


def store() -> LocalStore | S3Store:
    s = get_settings()
    return _store(str(s.media_path), s.media_s3_endpoint, s.media_s3_bucket, s.media_s3_access_key,
                  s.media_s3_secret_key, s.media_s3_region)


def temp_path(suffix: str = "") -> Path:
    """A temporary file for a download before it goes to the store."""
    f = tempfile.NamedTemporaryFile(prefix="loom-", suffix=suffix, delete=False)
    f.close()
    return Path(f.name)
