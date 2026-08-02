"""Datasette plugin: serve original QC images by qc_results row id (read-only).

Security model:
- Only a numeric rowid is accepted; the file path comes from the database
  record, never from user input, so there is no path-traversal surface.
- Missing record / missing file -> 404, no directory information leaked.

NOTE: keep this file ASCII-only. Datasette's plugins-dir loader reads
source files with the platform default encoding (GBK on zh-CN Windows),
so non-ASCII characters would crash plugin loading. Chinese docs live in
core/qc_dashboard.py (loaded via normal UTF-8 imports).
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

from datasette import hookimpl
from datasette.utils.asgi import Response

_MAX_IMAGE_BYTES = 60 * 1024 * 1024  # 60MB cap, covers 5120x5120 BMP


@hookimpl
def register_routes():
    return [
        (r"/-/qc-img/(?P<rowid>\d+)", qc_image),
    ]


async def qc_image(datasette, request):
    rowid = request.url_vars["rowid"]
    try:
        db = datasette.get_database("visionocr")
    except KeyError:
        return Response.text("visionocr database not found", status=404)
    result = await db.execute(
        "SELECT image_path FROM qc_results WHERE id = ?", [rowid]
    )
    if not result.rows:
        return Response.text("record not found", status=404)
    image_path = result.rows[0][0] or ""
    p = Path(image_path)
    if not image_path or not p.is_file():
        return Response.text("image file missing (moved?)", status=404)
    if p.stat().st_size > _MAX_IMAGE_BYTES:
        return Response.text("image too large", status=413)
    data = p.read_bytes()
    ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    return Response(
        data,
        content_type=ctype,
        headers={"Cache-Control": "public, max-age=3600"},
    )
