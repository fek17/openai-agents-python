# Car Park Detection with OpenAI Agents SDK

This example demonstrates how to use the OpenAI Agents SDK with vision capabilities to detect car parking spaces in images and export bounding box data to CSV format.

## Overview

The scripts use GPT-4o's vision capabilities through the Agents SDK to:
1. Analyze parking lot images
2. Detect individual parking spaces
3. Generate bounding boxes for each space
4. Export results to CSV with coordinates and dimensions

## Files

- `car_park_detection.py` - Full-featured script with detailed documentation
- `car_park_detection_simple.py` - Simple command-line tool for quick use

## Prerequisites

1. Python 3.9 or newer
2. OpenAI Agents SDK installed: `pip install openai-agents`
3. OpenAI API key set as environment variable: `export OPENAI_API_KEY='your-key-here'`

## Quick Start

### Simple Usage (Command Line)

```bash
# Basic usage
uv run python car_park_detection_simple.py parking_lot.jpg

# Specify output CSV file
uv run python car_park_detection_simple.py parking_lot.jpg results.csv
```

### Programmatic Usage

```python
import asyncio
from car_park_detection import process_image

async def main():
    # Process a local image
    await process_image(
        image_path="parking_lot.jpg",
        output_csv="parking_spaces.csv",
        is_url=False
    )

    # Or process an image from URL
    await process_image(
        image_path="https://example.com/parking.jpg",
        output_csv="parking_spaces.csv",
        is_url=True
    )

asyncio.run(main())
```

## Output Format

The CSV file contains the following columns:

| Column | Description |
|--------|-------------|
| `image_name` | Name of the analyzed image |
| `space_id` | Sequential ID for each parking space |
| `x` | X coordinate of top-left corner (normalized 0-1) |
| `y` | Y coordinate of top-left corner (normalized 0-1) |
| `width` | Width of bounding box (normalized 0-1) |
| `height` | Height of bounding box (normalized 0-1) |

### Example CSV Output

```csv
image_name,space_id,x,y,width,height
parking_lot.jpg,1,0.1234,0.2345,0.0567,0.0890
parking_lot.jpg,2,0.2345,0.2345,0.0567,0.0890
parking_lot.jpg,3,0.3456,0.2345,0.0567,0.0890
```

## Coordinate System

Coordinates are normalized to the 0-1 range:

```
(0,0) ──────────────────> x (1,0)
  │
  │    ┌─────────┐
  │    │ (x,y)   │
  │    │   ┌─────┼─┐
  │    │   │     │h│
  │    └───┼─────┘ │
  │        │   w   │
  │        └───────┘
  ▼
  y
(0,1)
```

- `(x, y)` = top-left corner of bounding box
- `width` = horizontal span
- `height` = vertical span
- All values are fractions of image dimensions (0.0 to 1.0)

## How It Works

1. **Image Input**: The script accepts local files or URLs
2. **Vision Analysis**: Uses GPT-4o to analyze the image with high detail
3. **Detection**: AI identifies individual parking spaces (marked bays, lines, etc.)
4. **Structured Output**: Uses Pydantic models to get consistent, typed responses
5. **CSV Export**: Saves bounding boxes with normalized coordinates

## Key Features

- **Structured Output**: Uses Pydantic models for type-safe results
- **Vision Capabilities**: Leverages GPT-4o's image understanding
- **Flexible Input**: Supports local files and remote URLs
- **High Detail Mode**: Uses high-resolution analysis for better detection
- **Normalized Coordinates**: Scale-independent bounding boxes (0-1 range)

## Example Agent Instructions

The detection agent is instructed to:
- Identify all parking spaces (marked bays, line-delineated spots)
- Detect both occupied and empty spaces
- Provide precise normalized bounding boxes
- Return results in structured format

## Visualization (Optional)

To visualize the detected parking spaces, you can use the coordinates with image processing libraries:

```python
from PIL import Image, ImageDraw

def draw_boxes(image_path, csv_path, output_path):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row['x']) * img.width
            y = float(row['y']) * img.height
            w = float(row['width']) * img.width
            h = float(row['height']) * img.height

            draw.rectangle([x, y, x+w, y+h], outline='red', width=2)

    img.save(output_path)
    print(f"Annotated image saved to: {output_path}")

# Usage
draw_boxes('parking_lot.jpg', 'parking_spaces.csv', 'annotated.jpg')
```

## Limitations

- Detection accuracy depends on image quality and clarity
- Works best with clearly marked parking spaces
- May require manual verification for complex layouts
- API usage incurs costs based on OpenAI pricing

## Advanced Usage

See `car_park_detection.py` for:
- Detailed documentation
- Customizable agent instructions
- Error handling
- Extended functionality

## Support

For issues or questions about the OpenAI Agents SDK:
- Documentation: https://openai.github.io/openai-agents-python/
- Repository: https://github.com/openai/openai-agents-python
- Examples: https://github.com/openai/openai-agents-python/tree/main/examples
