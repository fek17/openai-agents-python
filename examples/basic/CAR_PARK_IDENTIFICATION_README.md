# Car Park Identification Agent

This example demonstrates how to use the OpenAI Agents SDK with vision capabilities to identify car parks (parking lots) in satellite or aerial images.

## Overview

The agent:
- 🎯 Identifies **entire car parks** (not individual parking spaces) from satellite/aerial images
- 📦 Returns structured output with bounding box coordinates
- 🎨 Draws bounding boxes on the image with labels
- 💾 Saves the annotated image for visual inspection

## Features

- Vision-based detection using GPT-4o
- **Structured Outputs** with Pydantic models for guaranteed schema adherence
- Normalized bounding box coordinates (0-1 range)
- Confidence scoring for each detection
- Visual annotation with colored bounding boxes
- Descriptive labels for each car park

## How It Works

This example uses OpenAI's **Structured Outputs** feature to ensure the agent returns data that exactly matches the defined schema. The agent:

1. Analyzes the satellite image using vision capabilities
2. Returns a `CarParksDetection` object with guaranteed schema adherence
3. The response includes structured bounding boxes, confidence levels, and descriptions
4. Python code then uses this structured data to draw annotations on the image

This approach separates concerns: the agent focuses on detection using structured outputs, while Python handles the image rendering.

## Requirements

```bash
pip install pillow
```

Or with uv:
```bash
uv pip install pillow
```

## Usage

### 1. Prepare a Satellite Image

You need a satellite or aerial image containing car parks. You can obtain these from:

**Free Sources:**
- **Google Earth**: Screenshot satellite views of areas with parking lots
- **Sentinel Hub**: https://www.sentinel-hub.com/
- **NASA Earth Observatory**: https://earthobservatory.nasa.gov/
- **USGS Earth Explorer**: https://earthexplorer.usgs.gov/
- **Bing Maps**: Aerial view screenshots

**Tips for good results:**
- Choose images with clear, visible car parks
- Higher resolution images work better
- Ensure good contrast between car parks and surroundings
- Shopping centers, stadiums, airports are good examples

### 2. Place Your Image

Save your satellite image to:
```
examples/basic/media/satellite_image.jpg
```

Or update the `IMAGE_PATH` variable in the script to point to your image location.

**Supported formats:** JPG, JPEG, PNG, GIF, BMP, WEBP
The script automatically detects the image format based on the file extension.

### 3. Run the Script

```bash
cd /home/user/openai-agents-python
uv run python examples/basic/car_park_identification.py
```

### 4. View Results

The script will:
1. Analyze the image for car parks
2. Print detection results to the console
3. Save an annotated image as `car_parks_annotated.jpg`

**Example Output:**
```
CAR PARK DETECTION RESULTS
============================================================

Total car parks detected: 3

Car Park #1:
  Confidence: high
  Description: Large rectangular parking lot with visible markings
  Bounding Box:
    x_min: 0.2341, y_min: 0.1523
    x_max: 0.4521, y_max: 0.3245

Car Park #2:
  Confidence: medium
  Description: Smaller parking area adjacent to building
  Bounding Box:
    x_min: 0.5234, y_min: 0.4123
    x_max: 0.6421, y_max: 0.5234
...

Annotated image saved to: examples/basic/car_parks_annotated.jpg
```

## Understanding the Output

### Bounding Box Coordinates

Coordinates are **normalized** (0-1 range):
- `x_min`: Left edge (0 = left side of image, 1 = right side)
- `y_min`: Top edge (0 = top of image, 1 = bottom)
- `x_max`: Right edge
- `y_max`: Bottom edge

To convert to pixel coordinates:
```python
pixel_x_min = x_min * image_width
pixel_y_min = y_min * image_height
```

### Confidence Levels
- **High**: Clear car park with visible markings and vehicles
- **Medium**: Likely car park but some ambiguity
- **Low**: Possible car park but uncertain

## Next Steps: Getting Lat/Long Coordinates

To get geographic coordinates (latitude/longitude) for the detected car parks, you'll need:

1. **GeoTIFF Images**: Use satellite images with embedded geospatial metadata
2. **GDAL Library**: For coordinate transformation
3. **Coordinate Transformation**: Convert image pixel coordinates to geographic coordinates

**Example approach:**
```python
from osgeo import gdal

# Open GeoTIFF.
dataset = gdal.Open('satellite_image.tif')
geotransform = dataset.GetGeoTransform()

# Convert pixel to geographic coordinates.
def pixel_to_latlong(x_pixel, y_pixel, geotransform):
    x_geo = geotransform[0] + x_pixel * geotransform[1]
    y_geo = geotransform[3] + y_pixel * geotransform[5]
    return x_geo, y_geo
```

## Customization

### Adjust Detection Instructions

Modify the agent instructions in `car_park_identification.py` to fine-tune detection:

```python
agent = Agent(
    name="CarParkDetector",
    instructions="""
    [Customize detection criteria here]
    - Minimum size requirements
    - Specific features to look for
    - Areas to ignore
    """,
    ...
)
```

### Change Visualization

Modify the `draw_bounding_boxes` function to:
- Change colors
- Adjust line thickness
- Add more annotations
- Change font size

### Use Different Models

Try other vision-capable models:
```python
agent = Agent(
    ...
    model="gpt-4o-mini",  # Faster, cheaper
    # or
    model="gpt-4-turbo",  # Alternative
)
```

## Troubleshooting

**Image not found error:**
- Verify the image path is correct
- Check file permissions
- Ensure the image file exists

**No car parks detected:**
- Try a different image with clearer car parks
- Adjust the detection instructions
- Use higher quality images
- Try `detail="high"` in the image input

**Poor detection quality:**
- Use higher resolution images
- Ensure good contrast
- Try different satellite imagery sources
- Adjust confidence thresholds in your logic

**Import errors:**
- Run `uv pip install pillow`
- Ensure you're using uv to run: `uv run python ...`

## Architecture

The example demonstrates several SDK features:

1. **Vision Input**: Loading and encoding local images with automatic format detection
2. **Structured Outputs**: Using Pydantic models to guarantee schema-compliant responses
   - The agent's `output_type` parameter ensures the response matches the `CarParksDetection` schema
   - No need for validation or retry logic - the output is guaranteed to match the schema
3. **Separation of Concerns**:
   - Agent handles detection and returns structured data
   - Python code handles visualization (drawing bounding boxes)
4. **Agent Instructions**: Guiding the model with specific detection criteria

## Related Examples

- `examples/basic/local_image.py`: Basic image loading
- `examples/basic/tools.py`: Function tool creation
- `examples/basic/non_strict_output_type.py`: Structured outputs

## License

This example is part of the OpenAI Agents SDK and follows the same license.
