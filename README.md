🚗 Car Wash Vehicle Counter
This is a lightweight, real-time vehicle detection and counting system tailored for Car Wash businesses. It uses the YOLOv8n model and a custom IoU-based tracker to detect and track cars and motorcycles entering a predefined region of interest (ROI). The system includes daily SMS reporting via Twilio and is designed for ease of deployment and minimal resource consumption.

📦 Features
✅ Real-time vehicle detection (cars and motorcycles)

✅ Custom IoU tracker with unique ID assignment

✅ Entry line logic for accurate vehicle counting

✅ Daily SMS reports using Twilio API

✅ Lightweight architecture with YOLOv8n

✅ Support for custom video input

✅ Ready-to-use Python pipeline

🔜 OCR placeholder for future license plate recognition

📁 Project Structure
php
Copiar
Editar
📦 car-wash-vehicle-counter
├── main.py                # Main entry point for the video processing pipeline
├── tracker.py             # IoU-based tracking logic
├── utils.py               # Helper functions
├── config.py              # Model and Twilio configuration
├── static/                # Input and output videos
├── templates/             # HTML templates (if web interface is added)
├── requirements.txt       # Python dependencies
└── README.md              # This file
🛠️ Installation
Clone the repository

bash
Copiar
Editar
git clone https://github.com/your-username/car-wash-vehicle-counter.git
cd car-wash-vehicle-counter
Create virtual environment (optional but recommended)

bash
Copiar
Editar
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install dependencies

bash
Copiar
Editar
pip install -r requirements.txt
Download YOLOv8n model

bash
Copiar
Editar
pip install ultralytics
yolo task=detect mode=predict model=yolov8n.pt
🧪 Usage
Run the vehicle detection and counting system
bash
Copiar
Editar
python main.py --video path/to/your/video.mp4
You can modify the ROI, entry line, and object classes to detect inside main.py.

📲 Twilio SMS Integration
To receive daily SMS reports:

Set up a Twilio account at twilio.com

Add your credentials in config.py:

python
Copiar
Editar
TWILIO_ACCOUNT_SID = 'your_account_sid'
TWILIO_AUTH_TOKEN = 'your_auth_token'
TWILIO_PHONE_FROM = '+1234567890'
TWILIO_PHONE_TO = '+0987654321'
The system sends a summary of detected vehicles after processing.

🎯 Future Work
Add OCR for license plate recognition

Deploy web interface using Flask

Improve performance under low light or occlusion

Optimize model for edge devices (e.g., Raspberry Pi, Jetson Nano)

📜 License
This project is licensed under the MIT License. Feel free to use, modify, and contribute.

🤝 Contributing
Pull requests are welcome! If you encounter issues or want to request a feature, open an issue on GitHub.

📷 Demo

✉️ Contact
For questions, collaboration, or deployment inquiries, feel free to reach out via the Issues tab or email.
