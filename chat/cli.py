"""Interactive chat CLI: run a modality's model once, then ask free-form
natural-language questions about the result, in whatever language you ask.

Usage:
    python -m chat.cli --input sample_xray.png --modality xray
"""

import argparse

from chat.assistant import ChatSession


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the input file.")
    parser.add_argument("--modality", required=True, help="xray, ct, mri, ultrasound, or ecg.")
    args = parser.parse_args()

    session = ChatSession(args.modality, args.input)
    pred = session.prediction
    print(f"[{args.modality}] prediction: {pred['label']} (confidence {pred['confidence']:.2f})")
    print("Ask anything about this result. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("you> ").strip()
        except EOFError:
            break
        if question.lower() in ("exit", "quit", ""):
            break
        answer = session.ask(question)
        print(f"assistant> {answer}\n")


if __name__ == "__main__":
    main()
