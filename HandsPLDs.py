import os
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm  # Import tqdm library for progress bar

class HandTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HandPLDs Tracking App")

        self.video_paths = []
        self.smoothing_factor = tk.DoubleVar()
        self.output_size = tk.StringVar(value="1920x1080")
        self.use_same_resolution = tk.BooleanVar()
        self.num_hands = tk.IntVar(value=1)
        self.dots_size = tk.IntVar(value=5)

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Select Video(s):").grid(row=0, column=0, sticky='w', padx=10, pady=5)
        tk.Button(self.root, text="Browse", command=self.browse_videos).grid(row=0, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Smoothing Factor:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
        tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient="horizontal", variable=self.smoothing_factor).grid(row=1, column=1, padx=10, pady=5)

        tk.Label(self.root, text="Output Video Size:").grid(row=2, column=0, sticky='w', padx=10, pady=5)
        tk.Entry(self.root, textvariable=self.output_size).grid(row=2, column=1, padx=10, pady=5)

        tk.Checkbutton(self.root, text="Use Same Resolution as Input", variable=self.use_same_resolution, command=self.set_output_size).grid(row=3, columnspan=2, padx=10, pady=5)

        tk.Label(self.root, text="Number of Hands to Track:").grid(row=4, column=0, sticky='w', padx=10, pady=5)
        tk.Radiobutton(self.root, text="One", variable=self.num_hands, value=1).grid(row=4, column=1, padx=10, pady=5, sticky='w')
        tk.Radiobutton(self.root, text="Two", variable=self.num_hands, value=2).grid(row=4, column=1, padx=10, pady=5, sticky='e')

        tk.Label(self.root, text="Dots Size:").grid(row=5, column=0, sticky='w', padx=10, pady=5)
        tk.Scale(self.root, from_=1, to=20, orient="horizontal", variable=self.dots_size).grid(row=5, column=1, padx=10, pady=5)

        self.processing_label = tk.Label(self.root, text="", fg="yellow")
        self.processing_label.grid(row=6, columnspan=2, padx=10, pady=10)

        tk.Button(self.root, text="Process Videos", command=self.process_videos).grid(row=7, columnspan=2, padx=10, pady=10)

    def browse_videos(self):
        self.video_paths = filedialog.askopenfilenames(title="Select Video(s)", filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
    
    def set_output_size(self):
        if self.use_same_resolution.get() and self.video_paths:
            cap = cv2.VideoCapture(self.video_paths[0])
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.output_size.set(f"{width}x{height}")

    def process_videos(self):
        if not self.video_paths:
            messagebox.showwarning("Warning", "Please select at least one video.")
            return

        self.processing_label.config(text="Processing...", fg="yellow")
        self.root.update_idletasks()  # Update the GUI immediately after setting the text
        
        output_size = tuple(map(int, self.output_size.get().split('x')))
        smoothing_factor = self.smoothing_factor.get()
        num_hands = self.num_hands.get()
        dots_size = self.dots_size.get()

        mp_hand = mp.solutions.hands
        hands = mp_hand.Hands(max_num_hands=num_hands)

        for video_path in self.video_paths:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            video_dir, video_file = os.path.split(video_path)
            output_file = os.path.splitext(video_file)[0] + "_output.mp4"
            output_path = os.path.join(video_dir, output_file)

            # Calculate the scaling factor to fit the entire video within the output dimensions
            scale_factor = min(output_size[0] / width, output_size[1] / height)

            scaled_width = int(width * scale_factor)
            scaled_height = int(height * scale_factor)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (scaled_width, scaled_height))

            previous_landmarks = [None] * num_hands

            progress_bar = tqdm(total=total_frames, desc=f"Processing {video_file}", unit="frames")
            while True:
                success, img = cap.read()
                if not success:
                    break

                black_image = np.zeros((scaled_height, scaled_width, 3), dtype=np.uint8)

                result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                if result.multi_hand_landmarks:
                    for i, hand_landmark in enumerate(result.multi_hand_landmarks):
                        if previous_landmarks[i] is not None:
                            for j in range(len(hand_landmark.landmark)):
                                hand_landmark.landmark[j].x = smoothing_factor * hand_landmark.landmark[j].x + (1 - smoothing_factor) * previous_landmarks[i].landmark[j].x
                                hand_landmark.landmark[j].y = smoothing_factor * hand_landmark.landmark[j].y + (1 - smoothing_factor) * previous_landmarks[i].landmark[j].y
                                hand_landmark.landmark[j].z = smoothing_factor * hand_landmark.landmark[j].z + (1 - smoothing_factor) * previous_landmarks[i].landmark[j].z

                        previous_landmarks[i] = hand_landmark

                        for id, landmark in enumerate(hand_landmark.landmark):
                            cx, cy = int(landmark.x * scaled_width), int(landmark.y * scaled_height)
                            cv2.circle(black_image, (cx, cy), dots_size, (255, 255, 255), cv2.FILLED)

                out.write(black_image)
                progress_bar.update(1)

            cap.release()
            out.release()
            progress_bar.close()

        self.processing_label.config(text="Video processing completed.", fg="green")

if __name__ == "__main__":
    root = tk.Tk()
    app = HandTrackingApp(root)
    root.mainloop()
