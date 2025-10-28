"""
Car Park Detection Example

This script uses the OpenAI Agents SDK with vision capabilities to detect
car parking spaces in images and export the bounding box data to CSV.
"""

import asyncio
import base64
import csv
import os
from pathlib import Path
from typing import List

from pydantic import BaseModel

from agents import Agent, Runner


class BoundingBox(BaseModel):
    """Represents a bounding box for a detected car parking space."""

    x: float  # X coordinate of top-left corner (normalized 0-1)
    y: float  # Y coordinate of top-left corner (normalized 0-1)
    width: float  # Width of the bounding box (normalized 0-1)
    height: float  # Height of the bounding box (normalized 0-1)


class ParkingSpaceDetection(BaseModel):
    """Structured output for parking space detection."""

    image_name: str
    parking_spaces: List[BoundingBox]
    total_spaces: int


def image_to_base64(image_path: str) -> str:
    """Convert a local image file to base64 encoding."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


async def detect_parking_spaces(
    image_path: str, is_url: bool = False
) -> ParkingSpaceDetection:
    """
    Detect parking spaces in an image using AI vision.

    Args:
        image_path: Path to local image file or URL
        is_url: Whether the image_path is a URL (True) or local file (False)

    Returns:
        ParkingSpaceDetection containing bounding boxes for each parking space
    """
    # Create agent with vision capabilities
    agent = Agent(
        name="Parking Space Detector",
        instructions="""You are an expert at detecting car parking spaces in images.

Analyze the image and identify all individual car parking spaces/bays.
For each parking space you detect, provide a bounding box with normalized coordinates.

Coordinates should be normalized (0-1 range):
- x: horizontal position of top-left corner (0 = left edge, 1 = right edge)
- y: vertical position of top-left corner (0 = top edge, 1 = bottom edge)
- width: width of the box (0-1 scale)
- height: height of the box (0-1 scale)

Be precise and detect all visible parking spaces, including:
- Marked parking bays
- Parking spots delineated by lines
- Both occupied and empty spaces

Return the results in the specified structured format.""",
        output_type=ParkingSpaceDetection,
        model="gpt-4o",  # Using GPT-4o for vision capabilities
    )

    # Get image name
    image_name = os.path.basename(image_path)

    # Prepare image content
    if is_url:
        image_content = {
            "type": "input_image",
            "detail": "high",  # Use high detail for better detection
            "image_url": image_path,
        }
    else:
        b64_image = image_to_base64(image_path)
        image_content = {
            "type": "input_image",
            "detail": "high",
            "image_url": f"data:image/jpeg;base64,{b64_image}",
        }

    # Run detection
    result = await Runner.run(
        agent,
        [
            {"role": "user", "content": [image_content]},
            {
                "role": "user",
                "content": f"Detect all car parking spaces in this image. The image name is: {image_name}",
            },
        ],
    )

    return result.final_output


def save_to_csv(detection: ParkingSpaceDetection, output_path: str) -> None:
    """
    Save parking space detection results to a CSV file.

    Args:
        detection: ParkingSpaceDetection object with bounding box data
        output_path: Path where the CSV file should be saved
    """
    with open(output_path, "w", newline="") as csvfile:
        fieldnames = [
            "image_name",
            "parking_space_id",
            "x",
            "y",
            "width",
            "height",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for idx, bbox in enumerate(detection.parking_spaces, start=1):
            writer.writerow(
                {
                    "image_name": detection.image_name,
                    "parking_space_id": idx,
                    "x": bbox.x,
                    "y": bbox.y,
                    "width": bbox.width,
                    "height": bbox.height,
                }
            )

    print(f"CSV saved to: {output_path}")
    print(f"Total parking spaces detected: {detection.total_spaces}")


async def process_image(image_path: str, output_csv: str, is_url: bool = False) -> None:
    """
    Process an image to detect parking spaces and save results.

    Args:
        image_path: Path to local image file or URL
        output_csv: Path where the output CSV should be saved
        is_url: Whether the image_path is a URL
    """
    print(f"Processing image: {image_path}")
    print("Detecting parking spaces...")

    # Detect parking spaces
    detection = await detect_parking_spaces(image_path, is_url=is_url)

    # Save to CSV
    save_to_csv(detection, output_csv)

    # Print summary
    print("\nDetection Summary:")
    print(f"Image: {detection.image_name}")
    print(f"Total parking spaces found: {detection.total_spaces}")
    print(f"\nBounding boxes saved to: {output_csv}")


async def main():
    """Example usage of the car park detection system."""
    # Example 1: Process a local image
    # Uncomment and modify the path to your image
    # await process_image(
    #     image_path="path/to/your/parking_lot_image.jpg",
    #     output_csv="parking_spaces.csv",
    #     is_url=False
    # )

    # Example 2: Process an image from a URL
    # This is a sample parking lot image URL (replace with your actual image)
    sample_url = "https://example.com/parking_lot.jpg"

    print("=" * 60)
    print("Car Parking Space Detection")
    print("=" * 60)
    print()
    print("Usage:")
    print("1. Set your OPENAI_API_KEY environment variable")
    print("2. Modify the image_path in the code to your image location")
    print("3. Run the script to detect parking spaces")
    print()
    print("The script will:")
    print("  - Analyze the image using AI vision")
    print("  - Detect all car parking spaces")
    print("  - Save bounding box data to a CSV file")
    print()
    print("=" * 60)
    print()
    print("To use this script, uncomment one of the examples in main()")
    print("and provide your image path or URL.")

    # Uncomment to process a URL image:
    # await process_image(
    #     image_path=sample_url,
    #     output_csv="parking_spaces_from_url.csv",
    #     is_url=True
    # )


if __name__ == "__main__":
    asyncio.run(main())
