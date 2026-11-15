import argparse
import ctypes
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO, solutions


WINDOW_NAME = "YOLOv11 - Live head count"


DEFAULT_MODEL = (
    Path(__file__).resolve().parent
    / "runs"
    / "detect"
    / "train"
    / "weights"
    / "best.pt"
)


def fit_frame_to_screen(frame):
    screen_width = ctypes.windll.user32.GetSystemMetrics(0)
    screen_height = ctypes.windll.user32.GetSystemMetrics(1)
    max_width = int(screen_width * 0.85)
    max_height = int(screen_height * 0.8)
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    display_size = (int(width * scale), int(height * scale))
    if scale < 1:
        return cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
    return frame


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
        default="cpu",
        help="Thiet bi suy luan: 0 cho GPU dau tien, hoac cpu.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--count-mode",
        choices=("occupancy", "crossing"),
        default="occupancy",
        help="occupancy hien thi dau thay va muc cao nhat; crossing dem luot qua vach.",
    )
    parser.add_argument(
        "--line-y",
        type=float,
        default=0.55,
        help="Vi tri vach theo ty le chieu cao hinh, tu 0.0 den 1.0.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Luu video da ve tracking va thong tin dem.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Duong dan file MP4 dau ra; khai bao tuy chon nay cung bat luu video.",
    )
    return parser.parse_args()


def get_output_path(source: int | str, output: Path | None) -> Path:
    if output is not None:
        output_path = output.expanduser().resolve()
        if output_path.suffix.lower() != ".mp4":
            raise ValueError("File dau ra can co duoi .mp4.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    source_name = Path(source).stem if isinstance(source, str) else "webcam"
    output_dir = Path(__file__).resolve().parent / "runs" / "detect" / "count"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_name}_count.mp4"
    suffix = 2
    while output_path.exists():
        output_path = output_dir / f"{source_name}_count{suffix}.mp4"
        suffix += 1
    return output_path


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.line_y <= 1.0:
        raise ValueError("--line-y phai nam trong khoang tu 0.0 den 1.0.")

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

    model = None
    counter = None
    writer = None
    max_visible_count = 0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            if model is None and counter is None:
                height, width = frame.shape[:2]
                if args.count_mode == "crossing":
                    line_y = int(height * args.line_y)
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
                        line_width=1,
                        show_conf=False,
                    )
                else:
                    model = YOLO(str(model_path))

                if args.device == "0":
                    print(f"Dang dung GPU: {torch.cuda.get_device_name(0)}")
                print("Nhan Q hoac ESC tren cua so video de thoat.")
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                display_frame = fit_frame_to_screen(frame)
                cv2.resizeWindow(
                    WINDOW_NAME, display_frame.shape[1], display_frame.shape[0]
                )
                if args.save or args.output is not None:
                    output_path = get_output_path(source, args.output)
                    fps = camera.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        fps = 30.0
                    output_height, output_width = frame.shape[:2]
                    output_width -= output_width % 2
                    output_height -= output_height % 2
                    codec = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(output_path),
                        codec,
                        fps,
                        (output_width, output_height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Khong tao duoc file video dau ra: {output_path}"
                        )
                    print(f"Video ket qua se duoc luu tai: {output_path}")

            if args.count_mode == "crossing":
                result = counter(frame)
                current_count = len(counter.track_ids)
                count_label = f"In: {result.in_count}  Out: {result.out_count}"
                display_frame = result.plot_im
            else:
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    device=args.device,
                    classes=[0],
                    conf=args.conf,
                    imgsz=args.imgsz,
                    verbose=False,
                )
                detection = results[0]
                display_frame = detection.plot(
                    line_width=1,
                    font_size=10,
                )
                boxes = detection.boxes
                current_count = len(boxes) if boxes is not None else 0
                max_visible_count = max(max_visible_count, current_count)
                count_label = f"Max: {max_visible_count}"

            cv2.rectangle(
                display_frame,
                (10, 10),
                (min(frame.shape[1] - 10, 300), 42),
                (0, 0, 0),
                thickness=cv2.FILLED,
            )
            cv2.putText(
                display_frame,
                f"Now: {current_count}  {count_label}",
                (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(
                    display_frame[
                        : display_frame.shape[0] - display_frame.shape[0] % 2,
                        : display_frame.shape[1] - display_frame.shape[1] % 2,
                    ]
                )
            cv2.imshow(WINDOW_NAME, fit_frame_to_screen(display_frame))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
        if args.count_mode == "occupancy":
            print(f"So dau cao nhat phat hien trong mot khung hinh: {max_visible_count}")
    finally:
        camera.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
