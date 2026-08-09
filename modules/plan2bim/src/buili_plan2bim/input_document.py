from __future__ import annotations

from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class PreparedDrawing(BaseModel):
    """Auditable raster input prepared from an image or a selected PDF page."""

    model_config = ConfigDict(extra="forbid")

    source_path: str
    render_path: str
    source_kind: str
    page_number: int = Field(ge=1)
    page_count: int = Field(ge=1)
    width_px: int = Field(ge=1)
    height_px: int = Field(ge=1)


def _validated_image(source: Path) -> PreparedDrawing:
    with Image.open(source) as image:
        width, height = image.size
        image.verify()
    return PreparedDrawing(
        source_path=str(source),
        render_path=str(source),
        source_kind="raster_image",
        page_number=1,
        page_count=1,
        width_px=width,
        height_px=height,
    )


def prepare_drawing(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    page_number: int = 1,
    pdf_dpi: int = 300,
) -> PreparedDrawing:
    """Validate an image or rasterize one PDF page without changing source provenance."""

    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if page_number < 1:
        raise ValueError("page_number must be at least 1")
    if pdf_dpi < 72 or pdf_dpi > 600:
        raise ValueError("pdf_dpi must be between 72 and 600")
    if source.suffix.lower() != ".pdf":
        if page_number != 1:
            raise ValueError("raster images contain one page; page_number must be 1")
        return _validated_image(source)

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - packaging guard
        raise RuntimeError("PDF input requires pypdfium2") from exc

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(str(source))
    page_count = len(document)
    if page_number > page_count:
        document.close()
        raise ValueError(f"page_number {page_number} exceeds PDF page count {page_count}")
    page = document[page_number - 1]
    bitmap = page.render(scale=pdf_dpi / 72.0)
    image = bitmap.to_pil().convert("RGB")
    render_path = destination / f"00-source-page-{page_number}.png"
    image.save(render_path, format="PNG", optimize=True)
    width, height = image.size
    bitmap.close()
    page.close()
    document.close()
    return PreparedDrawing(
        source_path=str(source),
        render_path=str(render_path),
        source_kind="raster_pdf",
        page_number=page_number,
        page_count=page_count,
        width_px=width,
        height_px=height,
    )
