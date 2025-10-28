"""Car Park Identification Test Script.

This script demonstrates the bounding box drawing functionality without requiring
an actual satellite image or API calls. It creates a mock image and mock detection
results to show how the annotation works.

This is useful for:
- Testing the drawing functionality
- Understanding the coordinate system
- Verifying your environment is set up correctly

Usage:
    uv run python examples/basic/car_park_identification_test.py
"""

import os

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field


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
    description: str = Field(description="Brief description of the car park")


class CarParksDetection(BaseModel):
    """Detection result containing all identified car parks."""

    car_parks: list[CarPark] = Field(description="List of detected car parks")
    total_count: int = Field(description="Total number of car parks detected")


def create_mock_satellite_image(width: int = 800, height: int = 600, output_path: str = "mock_satellite.jpg"):
    """Create a mock satellite-style image for testing."""
    # Create a new image with a gray-green background.
    img = Image.new("RGB", (width, height), color=(140, 160, 140))
    draw = ImageDraw.Draw(img)

    # Draw some "buildings" (dark gray rectangles).
    buildings = [
        (100, 100, 250, 200),  # Building 1.
        (300, 150, 450, 280),  # Building 2.
        (500, 300, 650, 450),  # Building 3.
    ]

    for building in buildings:
        draw.rectangle(building, fill=(80, 80, 90))

    # Draw some "roads" (dark lines).
    draw.rectangle([0, 250, 800, 270], fill=(60, 60, 60))  # Horizontal road.
    draw.rectangle([270, 0, 290, 600], fill=(60, 60, 60))  # Vertical road.

    # Draw some "car parks" (lighter rectangles with some lines for parking spaces).
    car_parks_areas = [
        (120, 220, 240, 350),  # Car park 1.
        (320, 300, 430, 420),  # Car park 2.
        (500, 100, 700, 250),  # Car park 3.
    ]

    for cp in car_parks_areas:
        # Draw the parking lot surface (light gray).
        draw.rectangle(cp, fill=(120, 120, 120))

        # Draw parking space lines.
        x1, y1, x2, y2 = cp
        # Draw vertical lines to simulate parking spaces.
        for x in range(x1 + 20, x2, 20):
            draw.line([(x, y1), (x, y2)], fill=(100, 100, 100), width=1)

    # Save the image.
    img.save(output_path)
    return output_path


def draw_bounding_boxes_on_image(
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
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        # Draw label.
        label = f"CP-{car_park.id} ({car_park.confidence})"
        # Draw label background.
        bbox_text = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1, y1 - 25), label, fill="white", font=font)

    # Save the annotated image.
    img.save(output_path)
    return f"Annotated image saved to: {output_path}"


def main():
    """Main function to test the car park identification visualization."""
    print("Car Park Identification - Test Script")
    print("=" * 60)

    # Create mock satellite image.
    print("\n1. Creating mock satellite image...")
    mock_image_path = os.path.join(os.path.dirname(__file__), "mock_satellite.jpg")
    create_mock_satellite_image(output_path=mock_image_path)
    print(f"   Mock image created: {mock_image_path}")

    # Create mock detection results.
    print("\n2. Creating mock detection results...")
    mock_detections = CarParksDetection(
        car_parks=[
            CarPark(
                id=1,
                bbox=BoundingBox(x_min=0.15, y_min=0.367, x_max=0.30, y_max=0.583),
                confidence="high",
                description="Large rectangular parking lot with visible markings, adjacent to building",
            ),
            CarPark(
                id=2,
                bbox=BoundingBox(x_min=0.40, y_min=0.50, x_max=0.538, y_max=0.70),
                confidence="high",
                description="Medium-sized parking area with clear parking space delineation",
            ),
            CarPark(
                id=3,
                bbox=BoundingBox(x_min=0.625, y_min=0.167, x_max=0.875, y_max=0.417),
                confidence="medium",
                description="Large parking lot with partial visibility, appears to be commercial",
            ),
        ],
        total_count=3,
    )

    print(f"   Total car parks: {mock_detections.total_count}")
    for cp in mock_detections.car_parks:
        print(f"   - Car Park #{cp.id}: {cp.confidence} confidence")

    # Draw bounding boxes.
    print("\n3. Drawing bounding boxes on image...")
    output_path = os.path.join(os.path.dirname(__file__), "mock_satellite_annotated.jpg")
    result = draw_bounding_boxes_on_image(mock_image_path, mock_detections, output_path)
    print(f"   {result}")

    # Print summary.
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  1. Mock satellite image: {mock_image_path}")
    print(f"  2. Annotated image:      {output_path}")
    print("\nNext steps:")
    print("  - Open the annotated image to see the bounding boxes")
    print("  - Try the full example with a real satellite image:")
    print("    uv run python examples/basic/car_park_identification.py")


if __name__ == "__main__":
    main()
