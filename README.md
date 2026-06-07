Image Caption Generator using ViT + Q-Former + Flan-T5
Project Overview
This project is a deep learning-based Image Caption Generator that automatically generates natural language descriptions for input images.
The architecture combines:
•	Vision Transformer (ViT) as the image encoder
•	Q-Former module for visual-language bridging
•	Flan-T5 Base as the language decoder
The model was trained on image-caption datasets and can generate meaningful captions for unseen images.
Dataset used : Flickcr32K
________________________________________
Model Architecture 
Image → ViT Encoder → Q-Former → Flan-T5 Decoder → Caption
Components
Vision Encoder
•	Google ViT Base Patch16 224
•	Frozen during training
Q-Former
•	Learnable Query Tokens
•	Multi-Head Cross Attention
•	Feed Forward Network
Language Decoder
•	Google Flan-T5 Base
•	Last decoder layers fine-tuned
________________________________________
Features
•	Upload an image through a Tkinter GUI
•	Generate captions automatically
•	Uses Transformer-based architecture
•	Supports JPG, JPEG, PNG and WEBP images
•	GPU acceleration when CUDA is available
________________________________________
Project Structure
Image-caption-generator/
│
├── app_vit47.py
├── model_vit47.py
├── requirements.txt
├── README.md
└── epoch_vit_4.pth
________________________________________
Installation
Clone Repository
git clone https://github.com/Abdulkareem-01/Image-caption-generator.git
cd Image-caption-generator
Install Dependencies
pip install -r requirements.txt
________________________________________
Download Model Weights
The trained model file is approximately 1.47 GB and is not included in the repository.
Download:
epoch_vit_4.pth
from the Google Drive link below:
https://drive.google.com/file/d/1DOJPRfecEXrg3zt_VUvsF_-gyLL2oyXn/view?usp=drive_link
Place the downloaded file inside the project directory:
Image-caption-generator/
├── app_vit47.py
├── model_vit47.py
└── epoch_vit_4.pth
________________________________________
Important Configuration
Open app_vit47.py and update:
CHECKPOINT_PATH = "epoch_vit_4.pth"
instead of using an absolute Windows path.
Current code:
CHECKPOINT_PATH = r"epoch_vit_4.pth"
Recommended:
CHECKPOINT_PATH = "epoch_vit_4.pth"
This makes the project portable across different computers.
________________________________________
Run Application
python app_vit47.py
________________________________________
Supported Image Formats
•	JPG
•	JPEG
•	PNG
•	WEBP
________________________________________
Technologies Used
•	Python
•	PyTorch
•	Transformers
•	Vision Transformer (ViT)
•	Flan-T5
•	Pillow
•	Tkinter
________________________________________
Future Improvements
•	Web-based interface using Flask or Gradio
•	Beam search tuning
•	Larger ViT models
•	BLEU, ROUGE and CIDEr evaluation
•	Deployment on Hugging Face Spaces
________________________________________
Author
Shaik Abdul Kareem
________________________________________
License
This project is provided for educational and research purposes.
