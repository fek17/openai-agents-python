"""Car Park Identification Agent.

This example demonstrates how to use an agent with vision capabilities to identify
car parks (parking lots) in satellite/aerial images and draw bounding boxes around them.

Requirements:
    pip install pillow

Usage:
    1. Place a satellite/aerial image in the examples/basic/media/ folder
    2. Update the IMAGE_PATH variable below to point to your image
    3. Run: uv run python examples/basic/car_park_identification.py

The script will:
    - Load the local image
    - Use a vision model to identify car parks
    - Return structured output with bounding box coordinates
    - Draw the bounding boxes on the image
    - Save the annotated image as 'car_parks_annotated.jpg'
"""

import asyncio
import base64
import mimetypes
import os

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from agents import Agent, Runner

# Path to your satellite/aerial image.
# Update this to point to your own satellite image with car parks.
# Supports: JPG, JPEG, PNG, GIF, BMP, WEBP (auto-detected from extension).
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "media/satellite_image.jpg")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "car_parks_annotated.jpg")


class BoundingBox(BaseModel):
    """Bounding box coordinates in normalized form (0-1 range)."""

    x_min: float = Field(description="Left x coordinate (0-1 range)")
    y_min: float = Field(description="Top y coordinate (0-1 range)")
    x_max: float = Field(description="Right x coordinate (0-1 range)")
    y_max: float = Field(description="Bottom y coordinate (0-1 range)")


class CarPark(BaseModel):
    """Represents a detected car park with its bounding box."""

    id: int = Field(description="Unique identifier for this car park")
    bbox: BoundingBox = Field(description="Bounding box coordinates")
    confidence: str = Field(description="Confidence level: high, medium, or low")
    description: str = Field(description="Brief description of the car park (e.g., size, features)")


class CarParksDetection(BaseModel):
    """Detection result containing all identified car parks."""

    car_parks: list[CarPark] = Field(description="List of detected car parks")
    total_count: int = Field(description="Total number of car parks detected")


def image_to_base64(image_path: str) -> tuple[str, str]:
    """Convert an image file to base64 string and detect MIME type.

    Returns:
        tuple: (base64_string, mime_type)
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    # Detect MIME type from file extension.
    mime_type, _ = mimetypes.guess_type(image_path)

    # Default to image/jpeg if detection fails.
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"

    return encoded_string, mime_type


def draw_bounding_boxes(
    image_path: str, detections: CarParksDetection, output_path: str
) -> str:
    """Draw bounding boxes on the image and save the result."""
    # Open the image.
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Try to use a nice font, fall back to default if not available.
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    # Draw each bounding box.
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]

    for i, car_park in enumerate(detections.car_parks):
        bbox = car_park.bbox

        # Convert normalized coordinates to pixel coordinates.
        x1 = int(bbox.x_min * width)
        y1 = int(bbox.y_min * height)
        x2 = int(bbox.x_max * width)
        y2 = int(bbox.y_max * height)

        # Choose color.
        color = colors[i % len(colors)]

        # Draw rectangle.
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label.
        label = f"CP-{car_park.id} ({car_park.confidence})"
        # Draw label background.
        bbox_text = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1, y1 - 25), label, fill="white", font=font)

    # Save the annotated image.
    img.save(output_path)
    return f"Annotated image saved to: {output_path}"


async def main():
    """Main function to run the car park identification agent."""
    # Check if image exists.
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        print("\nPlease provide a satellite/aerial image at the path specified.")
        print("You can download free satellite images from:")
        print("  - Google Earth")
        print("  - Sentinel Hub")
        print("  - NASA Earth Observatory")
        return

    # Convert image to base64 and detect format.
    print(f"Loading image from: {IMAGE_PATH}")
    b64_image, mime_type = image_to_base64(IMAGE_PATH)
    print(f"Detected image format: {mime_type}")

    # Create the agent with vision capabilities.
    agent = Agent(
        name="CarParkDetector",
        instructions="""You are a car park (parking lot) detection specialist.

Your task is to identify ENTIRE CAR PARKS (parking lots) in satellite/aerial images,
NOT individual parking spaces.

A car park is typically:
- A large paved area with visible parking spaces (lines/markings)
- Contains multiple rows of parked vehicles
- Usually rectangular or follows building contours
- May have entrance/exit points
- Often adjacent to commercial buildings, shopping centers, or public facilities

For each car park you detect:
1. Provide a bounding box with normalized coordinates (0-1 range)
2. Assign a confidence level (high/medium/low)
3. Add a brief description

Return your findings in the structured format specified.
""",
        output_type=CarParksDetection,
        model="gpt-4o",  # Use vision-capable model.
    )

    # Run the agent.
    print("\nAnalyzing image for car parks...")
    result = await Runner.run(
        agent,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "detail": "high",  # Use high detail for better detection.
                        "image_url": f"data:{mime_type};base64,{b64_image}",
                    }
                ],
            },
            {
                "role": "user",
                "content": """Analyze this satellite/aerial image and identify all car parks.

Remember: Identify ENTIRE car parks (parking lots), not individual parking spaces.""",
            },
        ],
    )

    # Print the detection results.
    print("\n" + "=" * 60)
    print("CAR PARK DETECTION RESULTS")
    print("=" * 60)

    if isinstance(result.final_output, CarParksDetection):
        detection = result.final_output
        print(f"\nTotal car parks detected: {detection.total_count}\n")

        for cp in detection.car_parks:
            print(f"Car Park #{cp.id}:")
            print(f"  Confidence: {cp.confidence}")
            print(f"  Description: {cp.description}")
            print(f"  Bounding Box:")
            print(f"    x_min: {cp.bbox.x_min:.4f}, y_min: {cp.bbox.y_min:.4f}")
            print(f"    x_max: {cp.bbox.x_max:.4f}, y_max: {cp.bbox.y_max:.4f}")
            print()

        # Draw bounding boxes.
        print("Drawing bounding boxes on image...")
        draw_result = draw_bounding_boxes(IMAGE_PATH, detection, OUTPUT_PATH)
        print(f"\n{draw_result}")
        print("\nNext steps:")
        print("  - View the annotated image to verify detections")
        print("  - For geolocation, you'll need to add coordinate transformation")
        print("    based on the image metadata (GeoTIFF tags, etc.)")

    else:
        print("Unexpected output format:")
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
