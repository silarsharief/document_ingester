class ValidationAgent:
    def validate(self, element_type: str, confidence: float, analysis_text: str) -> dict:
        """
        Decides if a visual element is worth keeping.
        """
        decision = {
            "is_valid": True,
            "reason": "Pass"
        }

        # Rule 1: Visual Confidence Check (YOLO)
        # We are lenient with tables (0.4) but strict with figures (0.5)
        threshold = 0.4 if element_type == "table" else 0.5
        if confidence < threshold:
            return {"is_valid": False, "reason": f"Low YOLO Confidence ({confidence:.2f})"}

        # Rule 2: Semantic Check (Gemini)
        bad_keywords = ["irrelevant", "cannot interpret", "unable to identify", "logo", "blank image"]
        if any(kw in analysis_text.lower() for kw in bad_keywords):
            return {"is_valid": False, "reason": "Gemini deemed irrelevant"}

        # Rule 3: Content Length Check
        if len(analysis_text) < 15:
             return {"is_valid": False, "reason": "Analysis too short/empty"}

        # Rule 4: Failure Check
        if "Analysis Failed" in analysis_text:
             return {"is_valid": False, "reason": "Vision API Failure"}

        return decision

validator = ValidationAgent()