"""
Simple Car Park Detection Example

Quick start example for detecting parking spaces in images.
"""

import asyncio
import base64
import csv
import sys
from pathlib import Path
from typing import List

from pydantic import BaseModel

from agents import Agent, Runner


class BoundingBox(BaseModel):
    """Bounding box with normalized coordinates (0-1 range)."""

    x: float
    y: float
    width: float
    height: float


class ParkingSpaceDetection(BaseModel):
    """Detection results."""

    image_name: str
    parking_spaces: List[BoundingBox]
    total_spaces: int


async def detect_parking_spaces(image_path: str) -> ParkingSpaceDetection:
    """Detect parking spaces in an image."""

    # Create vision agent
    agent = Agent(
        name="Parking Detector",
        instructions="""Detect all car parking spaces in the image.

For each parking space, provide a bounding box with normalized coordinates (0-1):
- x, y: top-left corner position
- width, height: box dimensions

Detect all visible parking bays, marked by lines, both empty and occupied.""",
        output_type=ParkingSpaceDetection,
        model="gpt-4o",
    )

    # Load and encode image
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    image_name = Path(image_path).name

    # Run detection
    result = await Runner.run(
        agent,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "detail": "high",
                        "image_url": f"data:image/jpeg;base64,{b64_image}",
                    }
                ],
            },
            {
                "role": "user",
                "content": f"Detect all parking spaces. Image name: {image_name}",
            },
        ],
    )

    return result.final_output


async def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python car_park_detection_simple.py <image_path> [output.csv]")
        print("\nExample:")
        print("  python car_park_detection_simple.py parking_lot.jpg")
        print("  python car_park_detection_simple.py parking_lot.jpg results.csv")
        return

    image_path = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "parking_spaces.csv"

    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        return

    print(f"Analyzing: {image_path}")
    print("This may take a moment...\n")

    # Detect parking spaces
    detection = await detect_parking_spaces(image_path)

    # Save to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "space_id", "x", "y", "width", "height"])

        for idx, bbox in enumerate(detection.parking_spaces, start=1):
            writer.writerow(
                [
                    detection.image_name,
                    idx,
                    f"{bbox.x:.4f}",
                    f"{bbox.y:.4f}",
                    f"{bbox.width:.4f}",
                    f"{bbox.height:.4f}",
                ]
            )

    # Print results
    print(f"✓ Detected {detection.total_spaces} parking spaces")
    print(f"✓ Results saved to: {output_csv}\n")

    print("Sample results:")
    for idx, bbox in enumerate(detection.parking_spaces[:3], start=1):
        print(
            f"  Space {idx}: x={bbox.x:.3f}, y={bbox.y:.3f}, "
            f"w={bbox.width:.3f}, h={bbox.height:.3f}"
        )

    if detection.total_spaces > 3:
        print(f"  ... and {detection.total_spaces - 3} more spaces")


if __name__ == "__main__":
    asyncio.run(main())
