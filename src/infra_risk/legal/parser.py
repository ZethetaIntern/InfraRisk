"""
Contract Parser using LayoutLM
Extracts structured data from legal PDF documents
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import pdfplumber

from src.infra_risk.utils.logging import get_logger

logger = get_logger(__name__)


class ContractParser:
    """Parses legal contracts using LayoutLMv3"""
    
    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        logger.info("ContractParser initialized", model=model_name, device=str(self.device))
    
    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """Parse PDF contract and extract structured data"""
        logger.info("Parsing PDF", file=file_path)
        
        pages_data = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text with layout info
                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False,
                    use_text_flow=True,
                )
                
                # Extract tables
                tables = page.extract_tables()
                
                # Convert page to image for LayoutLM
                img = page.to_image(resolution=224).original
                
                # Prepare LayoutLM input
                page_data = self._process_page(img, words, tables, page_num)
                pages_data.append(page_data)
        
        # Combine all pages
        return {
            "file_path": file_path,
            "num_pages": len(pages_data),
            "pages": pages_data,
            "full_text": " ".join([p["text"] for p in pages_data]),
        }
    
    def _process_page(
        self,
        image: Image.Image,
        words: List[Dict],
        tables: List,
        page_num: int,
    ) -> Dict[str, Any]:
        """Process single page with LayoutLM"""
        
        # Prepare text and boxes for LayoutLM
        text_segments = []
        boxes = []
        
        for word in words:
            text_segments.append(word["text"])
            # Normalize box to 0-1000
            x0 = int(word["x0"] / image.width * 1000)
            y0 = int(word["top"] / image.height * 1000)
            x1 = int(word["x1"] / image.width * 1000)
            y1 = int(word["bottom"] / image.height * 1000)
            boxes.append([x0, y0, x1, y1])
        
        # Run LayoutLM
        encoding = self.processor(
            image,
            text_segments,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Get predictions
        predictions = outputs.logits.argmax(-1).cpu().numpy()[0]
        
        # Decode entities
        entities = self._decode_entities(text_segments, boxes, predictions)
        
        return {
            "page_num": page_num,
            "text": " ".join(text_segments),
            "entities": entities,
            "tables": tables,
            "word_boxes": list(zip(text_segments, boxes)),
        }
    
    def _decode_entities(
        self,
        words: List[str],
        boxes: List[List[int]],
        predictions: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Decode LayoutLM predictions into entities"""
        # LayoutLMv3 uses BIO tagging
        # This is simplified - real implementation would use label mapping
        entities = []
        
        current_entity = None
        for i, (word, box, pred) in enumerate(zip(words, boxes, predictions)):
            if pred == 1:  # B-ENTITY
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": word,
                    "box": box,
                    "label": "ENTITY",
                    "start": i,
                    "end": i,
                }
            elif pred == 2:  # I-ENTITY
                if current_entity:
                    current_entity["text"] += " " + word
                    current_entity["box"] = self._merge_boxes(current_entity["box"], box)
                    current_entity["end"] = i
            else:  # O
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def _merge_boxes(self, box1: List[int], box2: List[int]) -> List[int]:
        """Merge two bounding boxes"""
        return [
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3]),
        ]
    
    def extract_key_value_pairs(self, parsed_data: Dict) -> List[Dict[str, Any]]:
        """Extract key-value pairs from parsed document"""
        # This would use LayoutLM's relation extraction capabilities
        # Simplified version
        kv_pairs = []
        
        for page in parsed_data.get("pages", []):
            text = page.get("text", "")
            entities = page.get("entities", [])
            
            # Simple pattern matching for common fields
            patterns = {
                "party": r"(?:party|contractor|employer|client|owner)[\s:]+([A-Z][A-Za-z\s&.,]+)",
                "date": r"(?:date|dated)[\s:]+(\d{1,2}[\s/-]\w+[\s/-]\d{4})",
                "amount": r"(?:amount|sum|price|cost)[\s:]+([\$\£\€]\s*[\d,]+(?:\.\d{2})?)",
                "percentage": r"(\d+(?:\.\d+)?\s*%)",
            }
            
            for key, pattern in patterns.items():
                import re
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    kv_pairs.append({
                        "key": key,
                        "value": match.strip(),
                        "page": page["page_num"],
                    })
        
        return kv_pairs