from pydantic import BaseModel, Field
from typing import Literal, Optional, List

class BoundingBox(BaseModel):
    page: int
    x1: float
    y1: float
    x2: float
    y2: float
    
    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

class DocElement(BaseModel):
    """The atomic unit of a document."""
    id: str = Field(..., description="Unique ID (e.g., page1_fig2)")
    type: Literal["text", "table", "figure", "header", "formula"]
    content: str = Field("", description="The Markdown text or empty if visual")
    
    # Context is crucial for your VLM requirement
    context_used: Optional[str] = Field(None, description="Surrounding text passed to VLM")
    
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    source: Literal["docling", "yolo_audit", "vlm_generated"]

class PageResult(BaseModel):
    page_number: int
    elements: List[DocElement]
    image_path: str  # Path to the debug image