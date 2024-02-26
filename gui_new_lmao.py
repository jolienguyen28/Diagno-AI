import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

import torch
import torchvision.transforms as transforms
from model import PneumoniaDetector   
class PneumoniaDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Pneumonia Detection App")
        self.master.geometry("{0}x{1}+0+0".format(self.master.winfo_screenwidth(), self.master.winfo_screenheight()))  # Set full screen
        self.model = PneumoniaDetector()
        self.model.load_state_dict(torch.load('model.pth')) 
        self.model.eval()

        self.master.configure(bg='#3498db')

        self.create_widgets()
        self.add_mission_text()  

    def create_widgets(self):
        # Frame for better organization
        main_frame = tk.Frame(self.master, padx=20, pady=20, bg='#3498db')
        main_frame.pack(expand=True, fill='both')

        # Heading
        heading_label = tk.Label(main_frame, text="Pneumonia Detection App", font=('Helvetica', 16, 'bold'), bg='#3498db', fg='white')
        heading_label.pack(pady=10)

        # Choose file label
        choose_label = tk.Label(main_frame, text="Choose an X-ray image", font=('Helvetica', 12), bg='#3498db', fg='white')
        choose_label.pack(pady=5)

        # Image display area
        self.image_label = tk.Label(main_frame, bg='#3498db')
        self.image_label.pack(pady=10)

        # Browse button
        browse_button = tk.Button(main_frame, text="Browse", command=self.browse_image, font=('Helvetica', 12), bg='#2980b9', fg='white')
        browse_button.pack(pady=10)

        # Detect button
        detect_button = tk.Button(main_frame, text="Detect Pneumonia", command=self.detect_pneumonia, font=('Helvetica', 12), bg='#2ecc71', fg='white')
        detect_button.pack(pady=10)

        # Result label
        self.result_label = tk.Label(main_frame, text="", font=('Helvetica', 14, 'bold'), bg='#3498db', fg='white')
        self.result_label.pack(pady=10)

    def add_mission_text(self):
        mission_text = """
        Our mission\n
        Welcome to DiagnoAI, your intelligent medical assistant revolutionizing healthcare through the power of deep learning and computer vision. 
        We are a dedicated team hailing from the Association of Computing Machinery (ACM) Organization Student Chapter at the University of South Florida, 
        where our passion for technology and innovation drives us to explore groundbreaking solutions in the field of healthcare.\n
        """

        # Mission label
        mission_label = tk.Label(self.master, text=mission_text, font=('Helvetica', 12), bg='#3498db', fg='white', justify='left')
        mission_label.pack(pady=10)

    def browse_image(self):
        self.image_path = filedialog.askopenfilename(title="Select an X-ray image",
                                                      filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if self.image_path:
            image = Image.open(self.image_path)
            image = image.resize((300, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.result_label.config(text="")

    def detect_pneumonia(self):
        if self.image_path:
            image = Image.open(self.image_path).convert('RGB')
            image = image.resize((224, 224), Image.LANCZOS)

            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0)

            with torch.no_grad():
                output = self.model(image_tensor)
                _, prediction = torch.max(output, 1)

            result_text = "No Pneumonia Detected" if prediction.item() == 0 else "Pneumonia Detected"
            self.result_label.config(text=result_text)
        else:
            self.result_label.config(text="Please select an image first")


if __name__ == "__main__":
    root = tk.Tk()
    app = PneumoniaDetectionApp(root)
    root.mainloop()
