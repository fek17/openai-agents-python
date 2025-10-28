"""Car Park Identification Agent with Three-Agent Quality Control Workflow.

This example demonstrates a multi-agent system with peer review:
- Agent 1 (Detector): Identifies car parks from satellite images
- Agent 2 (Reviewer): Evaluates the quality of detections
- Agent 3 (Corrector): Makes final corrections based on feedback

Requirements:
    pip install pillow

Usage:
    1. Place a satellite/aerial image in the examples/basic/media/ folder
    2. Update the IMAGE_PATH variable below to point to your image
    3. Run: uv run python examples/basic/car_park_identification.py

The script will:
    - Use Agent 1 to identify car parks
    - Draw initial bounding boxes
    - Use Agent 2 to review the detections
    - Use Agent 3 to make corrections
    - Export results to CSV
    - Save final annotated image
"""

import asyncio
import base64
import csv
import mimetypes
import os

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from agents import Agent, Runner

# Path to your satellite/aerial image.
# Update this to point to your own satellite image with car parks.
# Supports: JPG, JPEG, PNG, GIF, BMP, WEBP (auto-detected from extension).
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "media/satellite_image.jpg")
INITIAL_ANNOTATED_PATH = os.path.join(os.path.dirname(__file__), "car_parks_initial.jpg")
FINAL_ANNOTATED_PATH = os.path.join(os.path.dirname(__file__), "car_parks_final.jpg")
CSV_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "car_parks_detections.csv")


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


class CarParkIssue(BaseModel):
    """Represents an issue with a detected car park."""

    car_park_id: int = Field(description="ID of the car park with issues")
    issue_type: str = Field(
        description="Type of issue: 'false_positive', 'incorrect_boundary', 'missed_area', or 'ok'"
    )
    explanation: str = Field(description="Explanation of the issue")
    suggested_correction: str = Field(description="How to correct this issue")


class ReviewFeedback(BaseModel):
    """Agent 2's review of Agent 1's detections."""

    overall_quality: str = Field(
        description="Overall quality assessment: 'excellent', 'good', 'fair', or 'poor'"
    )
    issues: list[CarParkIssue] = Field(description="List of issues found with specific car parks")
    missed_car_parks: list[str] = Field(
        description="Descriptions of car parks that were missed entirely"
    )
    general_feedback: str = Field(description="General feedback and observations")


class CorrectionAction(BaseModel):
    """Represents a correction to be made."""

    action_type: str = Field(
        description="Type of action: 'remove', 'modify', 'add', or 'keep'"
    )
    car_park_id: int | None = Field(
        description="ID of car park to modify/remove, null for 'add' actions"
    )
    new_bbox: BoundingBox | None = Field(description="New bounding box (for modify/add actions)")
    confidence: str | None = Field(description="Confidence level (for modify/add actions)")
    description: str | None = Field(description="Description (for modify/add actions)")
    reason: str = Field(description="Reason for this correction")


class CorrectedDetections(BaseModel):
    """Agent 3's corrected detections."""

    corrections: list[CorrectionAction] = Field(description="List of correction actions to apply")
    final_car_parks: list[CarPark] = Field(description="Final list of corrected car parks")
    total_count: int = Field(description="Total number of car parks after corrections")
    correction_summary: str = Field(description="Summary of changes made")


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
    image_path: str,
    detections: CarParksDetection,
    output_path: str,
    title: str = "",
) -> str:
    """Draw bounding boxes on the image and save the result."""
    # Open the image.
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Try to use a nice font, fall back to default if not available.
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # Draw title if provided.
    if title:
        title_bbox = draw.textbbox((10, 10), title, font=title_font)
        draw.rectangle(title_bbox, fill="black")
        draw.text((10, 10), title, fill="white", font=title_font)

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


def export_to_csv(
    detections: CarParksDetection,
    image_path: str,
    csv_path: str,
) -> str:
    """Export detections to CSV file.

    Args:
        detections: CarParksDetection object with all car parks
        image_path: Path to the original image file
        csv_path: Path where CSV should be saved

    Returns:
        Success message
    """
    filename = os.path.basename(image_path)

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "filename",
            "car_park_id",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "confidence",
            "description",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for car_park in detections.car_parks:
            writer.writerow(
                {
                    "filename": filename,
                    "car_park_id": car_park.id,
                    "x_min": car_park.bbox.x_min,
                    "y_min": car_park.bbox.y_min,
                    "x_max": car_park.bbox.x_max,
                    "y_max": car_park.bbox.y_max,
                    "confidence": car_park.confidence,
                    "description": car_park.description,
                }
            )

    return f"CSV exported to: {csv_path}"


async def main():
    """Main function to run the three-agent car park identification workflow."""
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
    print("=" * 80)
    print("MULTI-AGENT CAR PARK IDENTIFICATION WORKFLOW")
    print("=" * 80)
    print(f"\nLoading image from: {IMAGE_PATH}")
    b64_image, mime_type = image_to_base64(IMAGE_PATH)
    print(f"Detected image format: {mime_type}")

    # ==================== AGENT 1: DETECTOR ====================
    print("\n" + "=" * 80)
    print("STEP 1: AGENT 1 (DETECTOR) - Initial Car Park Identification")
    print("=" * 80)

    agent1 = Agent(
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

Be thorough but avoid false positives. It's better to be conservative.
""",
        output_type=CarParksDetection,
        model="gpt-4o",
    )

    print("\nAgent 1 analyzing image...")
    result1 = await Runner.run(
        agent1,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "detail": "high",
                        "image_url": f"data:{mime_type};base64,{b64_image}",
                    }
                ],
            },
            {
                "role": "user",
                "content": "Analyze this satellite/aerial image and identify all car parks. Remember: Identify ENTIRE car parks (parking lots), not individual parking spaces.",
            },
        ],
    )

    if not isinstance(result1.final_output, CarParksDetection):
        print("Error: Agent 1 did not return expected output format")
        return

    initial_detections = result1.final_output
    print(f"\n✓ Agent 1 detected {initial_detections.total_count} car parks")

    for cp in initial_detections.car_parks:
        print(f"  - CP-{cp.id}: {cp.confidence} confidence - {cp.description[:60]}...")

    # Draw initial detections.
    print("\nDrawing initial detections...")
    draw_bounding_boxes(
        IMAGE_PATH, initial_detections, INITIAL_ANNOTATED_PATH, "Agent 1: Initial Detection"
    )
    print(f"✓ Initial annotated image saved: {INITIAL_ANNOTATED_PATH}")

    # ==================== AGENT 2: REVIEWER ====================
    print("\n" + "=" * 80)
    print("STEP 2: AGENT 2 (REVIEWER) - Quality Review")
    print("=" * 80)

    # Load the annotated image.
    b64_annotated, _ = image_to_base64(INITIAL_ANNOTATED_PATH)

    # Format detections as text for the reviewer.
    detections_text = "AGENT 1 DETECTIONS:\n\n"
    for cp in initial_detections.car_parks:
        detections_text += f"Car Park {cp.id}:\n"
        detections_text += f"  Bounding Box: ({cp.bbox.x_min:.3f}, {cp.bbox.y_min:.3f}) to ({cp.bbox.x_max:.3f}, {cp.bbox.y_max:.3f})\n"
        detections_text += f"  Confidence: {cp.confidence}\n"
        detections_text += f"  Description: {cp.description}\n\n"

    agent2 = Agent(
        name="CarParkReviewer",
        instructions="""You are a quality control specialist reviewing car park detections.

Your task is to evaluate the accuracy and completeness of car park identifications.

Review criteria:
1. Are all identified car parks actually car parks?
2. Are the bounding boxes accurately placed?
3. Were any obvious car parks missed?
4. Is the confidence level appropriate?

For each issue you find:
- Specify the car park ID
- Classify the issue type
- Explain the problem
- Suggest how to correct it

Be thorough and constructive in your feedback.
""",
        output_type=ReviewFeedback,
        model="gpt-4o",
    )

    print("\nAgent 2 reviewing detections...")
    result2 = await Runner.run(
        agent2,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "detail": "high",
                        "image_url": f"data:{mime_type};base64,{b64_image}",
                    },
                    {
                        "type": "input_image",
                        "detail": "high",
                        "image_url": f"data:image/jpeg;base64,{b64_annotated}",
                    },
                ],
            },
            {
                "role": "user",
                "content": f"""Review these car park detections for quality and accuracy.

The first image is the original satellite image.
The second image shows the detections with bounding boxes drawn.

{detections_text}

Evaluate the detections and provide detailed feedback.""",
            },
        ],
    )

    if not isinstance(result2.final_output, ReviewFeedback):
        print("Error: Agent 2 did not return expected output format")
        return

    review = result2.final_output
    print(f"\n✓ Agent 2 review complete")
    print(f"  Overall Quality: {review.overall_quality}")
    print(f"  Issues Found: {len(review.issues)}")
    print(f"  Missed Car Parks: {len(review.missed_car_parks)}")
    print(f"\n  General Feedback: {review.general_feedback}")

    if review.issues:
        print("\n  Specific Issues:")
        for issue in review.issues:
            print(f"    - CP-{issue.car_park_id} ({issue.issue_type}): {issue.explanation[:60]}...")

    if review.missed_car_parks:
        print("\n  Missed Car Parks:")
        for missed in review.missed_car_parks:
            print(f"    - {missed[:80]}...")

    # ==================== AGENT 3: CORRECTOR ====================
    print("\n" + "=" * 80)
    print("STEP 3: AGENT 3 (CORRECTOR) - Apply Corrections")
    print("=" * 80)

    # Format review feedback for Agent 3.
    feedback_text = f"""REVIEW FEEDBACK FROM AGENT 2:

Overall Quality: {review.overall_quality}

Issues with Specific Car Parks:
"""
    for issue in review.issues:
        feedback_text += f"\nCar Park {issue.car_park_id}:\n"
        feedback_text += f"  Issue Type: {issue.issue_type}\n"
        feedback_text += f"  Explanation: {issue.explanation}\n"
        feedback_text += f"  Suggested Correction: {issue.suggested_correction}\n"

    if review.missed_car_parks:
        feedback_text += "\n\nMissed Car Parks:\n"
        for i, missed in enumerate(review.missed_car_parks, 1):
            feedback_text += f"{i}. {missed}\n"

    feedback_text += f"\n\nGeneral Feedback: {review.general_feedback}"

    agent3 = Agent(
        name="CarParkCorrector",
        instructions="""You are a car park detection correction specialist.

Your task is to take the initial detections and the reviewer's feedback, then produce
the final corrected set of car park detections.

You can:
- Remove false positives (action_type='remove')
- Modify incorrect bounding boxes (action_type='modify')
- Add missed car parks (action_type='add')
- Keep correct detections (action_type='keep')

For each correction:
1. Specify the action type
2. Provide the car park ID (or null for new additions)
3. Include new bounding box and details if needed
4. Explain your reasoning

Your final_car_parks list should contain all car parks after corrections are applied.
Ensure each car park has a unique ID (starting from 1).
""",
        output_type=CorrectedDetections,
        model="gpt-4o",
    )

    print("\nAgent 3 applying corrections...")
    result3 = await Runner.run(
        agent3,
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "detail": "high",
                        "image_url": f"data:{mime_type};base64,{b64_image}",
                    },
                    {
                        "type": "input_image",
                        "detail": "high",
                        "image_url": f"data:image/jpeg;base64,{b64_annotated}",
                    },
                ],
            },
            {
                "role": "user",
                "content": f"""Review the initial detections and the feedback, then provide corrected detections.

INITIAL DETECTIONS:
{detections_text}

{feedback_text}

Provide your corrections and the final list of car parks.""",
            },
        ],
    )

    if not isinstance(result3.final_output, CorrectedDetections):
        print("Error: Agent 3 did not return expected output format")
        return

    corrected = result3.final_output
    print(f"\n✓ Agent 3 corrections complete")
    print(f"  Corrections Made: {len(corrected.corrections)}")
    print(f"  Final Car Park Count: {corrected.total_count}")
    print(f"\n  Correction Summary: {corrected.correction_summary}")

    print("\n  Correction Actions:")
    for correction in corrected.corrections:
        action_desc = correction.action_type
        if correction.car_park_id:
            action_desc += f" CP-{correction.car_park_id}"
        print(f"    - {action_desc}: {correction.reason[:60]}...")

    # ==================== FINAL OUTPUT ====================
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    # Create final detections object.
    final_detections = CarParksDetection(
        car_parks=corrected.final_car_parks, total_count=corrected.total_count
    )

    print(f"\nFinal car parks detected: {final_detections.total_count}\n")
    for cp in final_detections.car_parks:
        print(f"Car Park #{cp.id}:")
        print(f"  Confidence: {cp.confidence}")
        print(f"  Description: {cp.description}")
        print(f"  Bounding Box:")
        print(f"    x_min: {cp.bbox.x_min:.4f}, y_min: {cp.bbox.y_min:.4f}")
        print(f"    x_max: {cp.bbox.x_max:.4f}, y_max: {cp.bbox.y_max:.4f}")
        print()

    # Draw final detections.
    print("Drawing final bounding boxes...")
    draw_bounding_boxes(IMAGE_PATH, final_detections, FINAL_ANNOTATED_PATH, "Final Detection")
    print(f"✓ Final annotated image saved: {FINAL_ANNOTATED_PATH}")

    # Export to CSV.
    print("\nExporting to CSV...")
    csv_result = export_to_csv(final_detections, IMAGE_PATH, CSV_OUTPUT_PATH)
    print(f"✓ {csv_result}")

    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print(f"  1. Initial detections: {INITIAL_ANNOTATED_PATH}")
    print(f"  2. Final detections:   {FINAL_ANNOTATED_PATH}")
    print(f"  3. CSV export:         {CSV_OUTPUT_PATH}")
    print("\nNext steps:")
    print("  - Compare initial vs final annotated images")
    print("  - Use the CSV to calculate pixel coordinates:")
    print("    pixel_x = x_min * image_width")
    print("    pixel_y = y_min * image_height")
    print("  - Convert pixel coordinates to lat/long using GeoTIFF metadata")


if __name__ == "__main__":
    asyncio.run(main())
