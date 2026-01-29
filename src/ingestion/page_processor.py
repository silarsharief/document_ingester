import os
from docling.datamodel.document import ConversionResult
from PIL import Image

class PageProcessor:
    @staticmethod
    def save_page_images(result: ConversionResult, output_dir: str = "data/processed_images") -> list[str]:
        """
        Extracts images of each page for the Vision Auditor.
        """
        os.makedirs(output_dir, exist_ok=True)
        image_paths = []
        
        # Iterate through pages
        for page_no, page in result.document.pages.items():
            image_path = os.path.join(output_dir, f"page_{page_no}.png")
            
            # ARCHITECTURE FIX: 
            # page.image is an ImageRef. We must retrieve the actual PIL Image from it.
            # Depending on Docling version, it's either .pil_image or .image
            # We check safely.
            if hasattr(page.image, "pil_image"):
                pil_image = page.image.pil_image
            else:
                # Fallback: In some versions, page.image IS the PIL image or acts like it
                pil_image = page.image

            if pil_image:
                # We do the resizing using PIL's standard resize, not a .scale() method
                # 2.0x scale for better YOLO detection
                new_size = (int(pil_image.width * 2), int(pil_image.height * 2))
                resized_image = pil_image.resize(new_size, Image.LANCZOS)
                
                # Save to disk
                with open(image_path, "wb") as f:
                    resized_image.save(f, format="PNG")
                
                image_paths.append(image_path)
            
        print(f"   ✅ Saved {len(image_paths)} page images to {output_dir}")
        return image_paths