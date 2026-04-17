"""
Quick smoke test for Command A Vision support.

Usage:
    export COHERE_API_KEY=your-key
    python libs/cohere/examples/vision_example.py
    python libs/cohere/examples/vision_example.py --image https://your-image-url.jpg
"""

import argparse

from langchain_core.messages import HumanMessage

from langchain_cohere import ChatCohere

DEFAULT_IMAGE_URL = "https://cohere.com/favicon-32x32.png"
VISION_MODEL = "command-a-vision-07-2025"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Command A Vision via langchain-cohere"
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE_URL,
        help="Image URL or data URI (base64) to send to the model",
    )
    parser.add_argument(
        "--prompt",
        default="What's in this image?",
        help="Text prompt to accompany the image",
    )
    args = parser.parse_args()

    llm = ChatCohere(model=VISION_MODEL)

    message = HumanMessage(
        content=[
            {"type": "text", "text": args.prompt},
            {"type": "image_url", "image_url": {"url": args.image}},
        ]
    )

    print(f"Model:  {VISION_MODEL}")
    print(f"Image:  {args.image[:80]}{'...' if len(args.image) > 80 else ''}")
    print(f"Prompt: {args.prompt}")
    print("-" * 60)

    response = llm.invoke([message])
    print(response.content)


if __name__ == "__main__":
    main()
