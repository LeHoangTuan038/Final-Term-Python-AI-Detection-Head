import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import solutions


DEFAULT_MODEL = (
    Path(__file__).resolve().parent
    / "runs"
    / "detect"
    / "train"
    / "weights"
    / "best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dem dau dang thay va luot di qua vach tren camera."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="So webcam (mac dinh 0), duong dan video, hoac URL stream.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Duong dan den file best.pt.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Thiet bi suy luan: 0 cho GPU dau tien, hoac cpu.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Khong tim thay model: {model_path}")

    if args.device == "0" and not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch chua nhan GPU CUDA. Kiem tra driver NVIDIA va PyTorch CUDA."
        )

    source: int | str = int(args.source) if args.source.isdecimal() else args.source
    if isinstance(source, int):
        camera = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        camera = cv2.VideoCapture(source)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Khong mo duoc nguon camera/video: {args.source}. "
            "Hay kiem tra camera va thu --source 1 neu webcam dung so khac."
        )

    counter = None
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            if counter is None:
                height, width = frame.shape[:2]
                line_y = int(height * 0.55)
                region = [
                    (int(width * 0.1), line_y),
                    (int(width * 0.9), line_y),
                ]
                counter = solutions.ObjectCounter(
                    model=str(model_path),
                    region=region,
                    classes=[0],
                    tracker="bytetrack.yaml",
                    device=args.device,
                    conf=args.conf,
                    imgsz=args.imgsz,
                )
                if args.device == "0":
                    print(f"Dang dung GPU: {torch.cuda.get_device_name(0)}")
                print("Nhan Q hoac ESC tren cua so video de thoat.")

            result = counter(frame)
            current_count = len(counter.track_ids)
            display_frame = result.plot_im

            cv2.rectangle(
                display_frame,
                (10, 10),
                (min(frame.shape[1] - 10, 660), 84),
                (0, 0, 0),
                thickness=cv2.FILLED,
            )
            cv2.putText(
                display_frame,
                f"Heads now: {current_count}",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"Crossed line - In: {result.in_count}  Out: {result.out_count}",
                (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("YOLOv11 - Live head count", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
