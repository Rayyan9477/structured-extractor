    async def _ensure_models_loaded(self) -> None:
        """Ensure all required models are loaded before extraction."""
        if self._models_loaded:
            return
        
        try:
            self.logger.info("Loading required models...")
            
            # Load OCR model
            await self.ocr_engine.load_models()
            
            # Load NuExtract model
            await self.nuextract_engine.load_model()
            
            self._models_loaded = True
            self.logger.info("All models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise RuntimeError(f"Failed to load required models: {e}")
