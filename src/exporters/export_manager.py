"""
Export Manager for Structured Data

Handles exporting extracted structured data to various formats.
"""

import os
import json
import csv
import asyncio
from typing import Dict, Any, List, Optional, Union, Set
import xml.dom.minidom
import xml.etree.ElementTree as ET

from src.core.config_manager import ConfigManager
from src.core.logger import get_logger
from src.core.data_schema import ExtractionResults


class ExportManager:
    """
    Manager for exporting extraction results to various formats.
    Supports JSON, CSV, XML, and other common formats.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the export manager.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Load export configuration
        self.export_config = config.get("export", {})
        self.supported_formats = {"json", "csv", "xml", "txt"}
        
        self.logger.info("Export manager initialized")
    
    async def export(
        self,
        results: ExtractionResults,
        output_path: str,
        output_format: Optional[str] = None
    ) -> None:
        """
        Export extraction results to specified format.
        
        Args:
            results: Extraction results to export
            output_path: Path to save output file
            output_format: Format for output file (json, csv, xml, txt)
                If None, will be determined from output_path extension
        """
        # Determine format from extension if not specified
        if not output_format:
            _, ext = os.path.splitext(output_path)
            output_format = ext.lstrip('.').lower()
        else:
            output_format = output_format.lower()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Validate format
        if output_format not in self.supported_formats:
            self.logger.warning(
                f"Unsupported export format '{output_format}'. "
                f"Supported formats: {', '.join(self.supported_formats)}. "
                f"Defaulting to JSON."
            )
            output_format = "json"
        
        # Export based on format
        if output_format == "json":
            await self._export_to_json(results, output_path)
        elif output_format == "csv":
            await self._export_to_csv(results, output_path)
        elif output_format == "xml":
            await self._export_to_xml(results, output_path)
        elif output_format == "txt":
            await self._export_to_text(results, output_path)
        
        self.logger.info(f"Exported results to {output_path} in {output_format} format")
    
    async def _export_to_json(self, results: ExtractionResults, output_path: str) -> None:
        """
        Export results to JSON format.
        
        Args:
            results: Extraction results to export
            output_path: Path to save JSON file
        """
        try:
            # Convert results to dict
            results_dict = self._results_to_dict(results)
            
            # Run JSON export in a thread to avoid blocking
            await asyncio.to_thread(self._write_json, results_dict, output_path)
            
        except Exception as e:
            self.logger.error(f"JSON export failed: {e}", exc_info=True)
            raise
    
    def _write_json(self, data: Dict[str, Any], output_path: str) -> None:
        """
        Write data to JSON file.
        
        Args:
            data: Data to write
            output_path: Path to save file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    async def _export_to_csv(self, results: ExtractionResults, output_path: str) -> None:
        """
        Export results to CSV format.
        
        Args:
            results: Extraction results to export
            output_path: Path to save CSV file
        """
        try:
            # Flatten structured data for CSV
            flattened_data = self._flatten_dict(results.structured_data)
            
            # Add metadata columns
            flattened_data.update({
                "metadata_file_name": results.metadata.file_name,
                "metadata_file_path": results.metadata.file_path,
                "metadata_file_size": results.metadata.file_size,
                "metadata_page_count": results.metadata.page_count,
                "metadata_processing_time": results.metadata.processing_time,
                "metadata_extraction_date": results.metadata.extraction_date,
                "ocr_confidence": results.ocr_confidence,
                "extraction_confidence": results.extraction_confidence,
                "overall_confidence": results.overall_confidence
            })
            
            # Run CSV export in a thread
            await asyncio.to_thread(self._write_csv, flattened_data, output_path)
            
        except Exception as e:
            self.logger.error(f"CSV export failed: {e}", exc_info=True)
            raise
    
    def _write_csv(self, data: Dict[str, Any], output_path: str) -> None:
        """
        Write data to CSV file.
        
        Args:
            data: Flattened data to write
            output_path: Path to save file
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(data.keys())
            
            # Write data row
            writer.writerow(data.values())
    
    async def _export_to_xml(self, results: ExtractionResults, output_path: str) -> None:
        """
        Export results to XML format.
        
        Args:
            results: Extraction results to export
            output_path: Path to save XML file
        """
        try:
            # Convert results to dict
            results_dict = self._results_to_dict(results)
            
            # Run XML export in a thread
            await asyncio.to_thread(self._write_xml, results_dict, output_path)
            
        except Exception as e:
            self.logger.error(f"XML export failed: {e}", exc_info=True)
            raise
    
    def _write_xml(self, data: Dict[str, Any], output_path: str) -> None:
        """
        Write data to XML file.
        
        Args:
            data: Data to write
            output_path: Path to save file
        """
        root = ET.Element("ExtractionResults")
        
        # Add metadata
        metadata = ET.SubElement(root, "Metadata")
        for key, value in data.get("metadata", {}).items():
            elem = ET.SubElement(metadata, key)
            elem.text = str(value)
        
        # Add confidences
        confidences = ET.SubElement(root, "Confidences")
        for conf_type in ["ocr_confidence", "extraction_confidence", "overall_confidence"]:
            if conf_type in data:
                elem = ET.SubElement(confidences, conf_type)
                elem.text = str(data[conf_type])
        
        # Add structured data
        structured = ET.SubElement(root, "StructuredData")
        self._dict_to_xml(data.get("structured_data", {}), structured)
        
        # Format and write XML
        xml_str = ET.tostring(root, encoding="utf-8")
        dom = xml.dom.minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
    
    def _dict_to_xml(self, data: Dict[str, Any], parent_elem: ET.Element) -> None:
        """
        Convert dictionary to XML elements.
        
        Args:
            data: Dictionary to convert
            parent_elem: Parent XML element
        """
        for key, value in data.items():
            if value is None:
                elem = ET.SubElement(parent_elem, key)
                elem.text = "null"
            elif isinstance(value, dict):
                elem = ET.SubElement(parent_elem, key)
                self._dict_to_xml(value, elem)
            elif isinstance(value, list):
                elem = ET.SubElement(parent_elem, key)
                for i, item in enumerate(value):
                    item_elem = ET.SubElement(elem, "item")
                    item_elem.set("index", str(i))
                    if isinstance(item, dict):
                        self._dict_to_xml(item, item_elem)
                    else:
                        item_elem.text = str(item)
            else:
                elem = ET.SubElement(parent_elem, key)
                elem.text = str(value)
    
    async def _export_to_text(self, results: ExtractionResults, output_path: str) -> None:
        """
        Export results to plain text format.
        
        Args:
            results: Extraction results to export
            output_path: Path to save text file
        """
        try:
            # Format text output
            text_output = self._format_text_output(results)
            
            # Write to file
            await asyncio.to_thread(self._write_text, text_output, output_path)
            
        except Exception as e:
            self.logger.error(f"Text export failed: {e}", exc_info=True)
            raise
    
    def _format_text_output(self, results: ExtractionResults) -> str:
        """
        Format extraction results as human-readable text.
        
        Args:
            results: Extraction results to format
            
        Returns:
            Formatted text string
        """
        lines = [
            "EXTRACTION RESULTS",
            "=================",
            "",
            f"File: {results.metadata.file_name}",
            f"Date: {results.metadata.extraction_date}",
            f"Pages: {results.metadata.page_count}",
            f"Confidence: {results.overall_confidence:.2f}",
            "",
            "STRUCTURED DATA:",
            "---------------"
        ]
        
        # Format structured data
        self._format_dict_as_text(results.structured_data, lines, indent=0)
        
        return "\n".join(lines)
    
    def _format_dict_as_text(
        self,
        data: Dict[str, Any],
        lines: List[str],
        indent: int = 0
    ) -> None:
        """
        Format dictionary as text lines.
        
        Args:
            data: Dictionary to format
            lines: List to append formatted lines to
            indent: Current indentation level
        """
        indent_str = "  " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                self._format_dict_as_text(value, lines, indent + 1)
            elif isinstance(value, list):
                lines.append(f"{indent_str}{key}:")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        lines.append(f"{indent_str}  [{i}]:")
                        self._format_dict_as_text(item, lines, indent + 2)
                    else:
                        lines.append(f"{indent_str}  [{i}]: {item}")
            else:
                lines.append(f"{indent_str}{key}: {value}")
    
    def _write_text(self, text: str, output_path: str) -> None:
        """
        Write text to file.
        
        Args:
            text: Text to write
            output_path: Path to save file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    def _results_to_dict(self, results: ExtractionResults) -> Dict[str, Any]:
        """
        Convert ExtractionResults to dictionary.
        
        Args:
            results: Results to convert
            
        Returns:
            Dictionary representation of results
        """
        return {
            "metadata": {
                "file_name": results.metadata.file_name,
                "file_path": results.metadata.file_path,
                "file_size": results.metadata.file_size,
                "page_count": results.metadata.page_count,
                "processing_time": results.metadata.processing_time,
                "extraction_date": results.metadata.extraction_date
            },
            "ocr_confidence": results.ocr_confidence,
            "extraction_confidence": results.extraction_confidence,
            "overall_confidence": results.overall_confidence,
            "structured_data": results.structured_data
        }
    
    def _flatten_dict(
        self,
        data: Dict[str, Any],
        parent_key: str = "",
        separator: str = "_"
    ) -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            data: Dictionary to flatten
            parent_key: Current parent key
            separator: Separator for keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for key, value in data.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            
            if isinstance(value, dict) and value:
                items.extend(self._flatten_dict(value, new_key, separator).items())
            elif isinstance(value, list):
                # Convert list to string representation
                items.append((new_key, str(value)))
            else:
                items.append((new_key, value))
                
        return dict(items) 