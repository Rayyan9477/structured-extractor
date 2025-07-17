# Medical Superbill Extractor UI

A modern, attractive, and user-friendly web interface for the Medical Superbill Data Extraction System built with Streamlit.

## Features

- **Intuitive Interface**: Clean and modern UI with intuitive navigation
- **Interactive Visualization**: Visualize extracted data with interactive components
- **Real-time Validation**: Validate extracted data and get suggested fixes
- **Batch Processing**: Process multiple documents at once with progress tracking
- **Configurable Export**: Export data in multiple formats with customization options
- **Advanced Configuration**: Fine-tune extraction settings through the UI

## Getting Started

### Installation

Make sure you have installed all required dependencies:

```bash
pip install -r requirements.txt
```

### Running the UI

To start the Streamlit UI, run:

```bash
python run_ui.py
```

Or directly with Streamlit:

```bash
streamlit run ui/app.py
```

## UI Components

The UI consists of several main components:

1. **Single File Processing**: Upload and process individual files
2. **Batch Processing**: Process multiple files in one operation
3. **Configuration**: Customize extraction settings
4. **Documentation**: Access help and documentation

## Development

The UI is built with Streamlit and follows a component-based architecture:

- `ui/app.py`: Main application entry point
- `ui/components/`: Reusable UI components
- `ui/assets/`: Static assets (CSS, images)
- `ui/pages/`: Additional pages for multi-page applications

## Customization

The UI appearance can be customized by modifying:

- `ui/assets/style.css`: Main CSS styling
- Theme variables in the CSS file
