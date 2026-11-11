1. tạo file data\_local.yaml
2. copy code vào file:

path: "C:/Users/SikunZed/Downloads/Python AI/Dataset Head Detection/Roboflow"

train: train/images

val: valid/images

test: test/images

names:

0: head

3. mở terminal VScode copy code vào:

Set-Location "C:\Users\SikunZed\Downloads\Python AI\Dataset Head Detection\Roboflow"

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

python -m pip install ultralytics
\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
Sẽ xảy ra lỗi: Lỗi này cho thấy pip trong .venv đã bị hỏng một phần sau khi quá trình cài đặt bị ngắt hoặc không thể dùng tệp vì đang bị chương trình khác giữ. Hãy khôi phục pip trong môi trường này:

Python trong .venv: 3.14.4

python -m pip --version: chạy được với pip 26.0.1

python -m pip check: không phát hiện dependency bị hỏng

4. chạy lại terminal:

Set-Location "C:\Users\SikunZed\Downloads\Python AI\Dataset Head Detection\Roboflow"
.\.venv\Scripts\Activate.ps1

python -m pip install ultralytics

5. Cop code:

yolo detect train model=yolo11s.pt data=data\_local.yaml epochs=100 imgsz=640 batch=8 patience=20 workers=0

6. Khi đã chốt mô hình, đánh giá trên tập test một lần:

yolo detect val model="runs/detect/train/weights/best.pt" data=data\_local.yaml split=test

Nhưng đây là train trên CPU sẽ rất lâu nên chạy lệnh đổi qua GPU dùng Pytorch và bỏ Torch:

Set-Location "C:\Users\SikunZed\Downloads\Python AI\Dataset Head Detection\Roboflow"
>> .\.venv\Scripts\Activate.ps1
>> 
>> nvidia-smi
>> 
>> python -m pip uninstall -y torch torchvision torchaudio
>> python -m pip install --upgrade pip
>> python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

Lệnh kiểm tra:
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Không nhận GPU')"

7. Để thử dự đoán trên một video:
Thông số thiết bị đang train:
PyTorch: 2.14.1+cu126
CUDA build: 12.6
CUDA available: True
GPU: NVIDIA GeForce RTX 3060  

Lệnh test bằng 1 video:
yolo detect track model="runs/detect/train/weights/best.pt" source="video-thu.mp4" tracker=bytetrack.yaml device=0 show=True save=True

Chạy trực tiếp bằng Webcam:
yolo detect track model="runs/detect/train/weights/best.pt" source=0 tracker=bytetrack.yaml device=0 show=True


8. Các bước kết nối vào camera
 1.  Kết nối nguồn camera
Webcam cắm vào PC: nguồn là 0.
Camera IP hoặc đầu ghi CCTV/NVR: lấy địa chỉ RTSP từ ứng dụng hoặc cấu hình thiết bị. Dạng thường gặp:
rtsp://TEN_DANG_NHAP:MAT_KHAU@DIA_CHI_IP:554/DUONG_DAN_STREAM

Nếu dùng camera IP, thay source=0 bằng địa chỉ RTSP đặt trong dấu ngoặc kép. Lệnh này hiển thị nhận diện và ID theo dõi, nhưng chưa đếm người qua đường.

2. Chạy bằng lệnh trong terminal VS Code (webcam laptop mặc định là camera 0):
python dem_nguoi.py

Cửa sổ video hiển thị:
- Heads now: số đầu đang được phát hiện trong khung hình hiện tại.
- Crossed line - In/Out: số lượt tracking đi qua đường đếm ảo.

Đường đếm được đặt ngang ở khoảng 55% chiều cao khung hình. Nhấn Q hoặc ESC
trên cửa sổ video để thoát. Nếu webcam mặc định không mở được, thử camera số 1:
python dem_nguoi.py --source 1

Có thể dùng video để kiểm tra trước:
python dem_nguoi.py --source "video-thu.mp4"