# **Real-time Human Pose & Gesture Analysis**

> **A cutting-edge Computer Vision application for real-time human motion understanding.**

---

## **Project Overview**

This project represents a significant leap in real-time human analysis, leveraging **advanced deep learning models** from Google's MediaPipe framework. It transforms live webcam input into a rich stream of data, including precise body pose, detailed hand movements, calculated joint angles, and interpreted human gestures. Designed for robust performance, it showcases the power of modern computer vision for interactive applications.

---

## **Key Capabilities**

### **1. Comprehensive Human Tracking**

* **Real-time Pose Estimation:** Utilizes MediaPipe's highly optimized pose model to detect and track **33 distinct 3D keypoints** across the entire human body (e.g., shoulders, elbows, knees, hips, ankles). The output is a dynamic skeletal overlay on the live video feed.
* **Detailed Hand Landmark Detection:** Integrates MediaPipe's specialized hand tracking model, identifying **21 precise 3D landmarks for each hand** (including individual fingers and palm points). This enables granular analysis of hand gestures.

### **2. In-depth Motion Analysis**

* **Advanced Joint Angle Calculation:** Performs real-time trigonometric calculations to derive precise angles for critical body joints. Currently includes:
    * **Elbow Angles** (Right & Left)
    * **Shoulder Angles** (Right & Left)
    These angles are presented numerically in a clear, dedicated information box on the display.
* **Intelligent Gesture Recognition:** Implements heuristic-based logic to interpret common human gestures from the combined pose and hand landmark data. Recognized gestures include:
    * **Arms Crossed**
    * **Thumbs Up**
    * **Thumbs Down**
    * **Open Palm / High Five**
    The detected gesture is dynamically displayed in a separate, prominent information box.

### **3. Performance & Usability**

* **GPU Accelerated Inference:** Optimized to harness the power of modern GPUs (e.g., **NVIDIA RTX 3050**) for smooth, low-latency real-time processing of multiple concurrent AI models.
* **Robust & User-Friendly:** Features comprehensive error handling for webcam access and model loading, providing clear console feedback and ensuring a stable application experience.

---

## **Technologies Utilized**

* **Python 3.10.x:** The core programming language, specifically chosen for optimal compatibility with MediaPipe.
* **OpenCV (`cv2`):** Essential for camera interfacing, real-time video stream processing, and visual display annotations.
* **MediaPipe (`mediapipe`):** Google's cutting-edge framework providing pre-trained deep learning models for pose and hand estimation.
* **NumPy:** Fundamental for high-performance numerical operations and vector mathematics, crucial for angle calculations.
* **Math:** Python's built-in module for core trigonometric functions.

---

## **Setup & Installation**

To get this real-time analysis application running on your local machine, please follow these **detailed, step-by-step instructions**:

1.  ### **Create a New Project Directory**
    ```bash
    # Example: Create a new folder on your D: drive
    D:
    cd D:\
    mkdir pose-analysis-app
    cd pose-analysis-app
    ```
    *(Ensure your project folder is named `pose-analysis-app`)*

2.  ### **Install Python 3.10.x** (Crucial for MediaPipe Compatibility)
    * Download the **Python 3.10.x Windows installer (64-bit)** from [python.org/downloads/windows/](https://www.python.org/downloads/windows/).
    * **During installation, select "Customize installation"** and on the "Advanced Options" screen, **ensure "Add Python to environment variables" is checked**. Choose your D: drive for the installation location (e.g., `D:\Python\Python310`).

3.  ### **Set Up & Activate Virtual Environment**
    * Open a **new Command Prompt** window.
    * Navigate to your project directory (`D:\pose-analysis-app`).
    * *(Optional: Delete old `venv` if present: `rmdir /s /q venv`)*
    * Create the virtual environment using Python 3.10:
        ```bash
        "D:\Python\Python310\python.exe" -m venv venv
        ```
        *(Adjust path to `python.exe` if your Python 3.10 installation is different).*
    * Activate the virtual environment:
        ```bash
        .\venv\Scripts\activate
        ```
        *(Your prompt should display `(venv)` at the start of the line)*

4.  ### **Install Project Dependencies**
    ```bash
    pip install mediapipe opencv-python numpy
    ```

5.  ### **Place `main.py`**
    * Ensure your `main.py` script (containing the project's code) is placed directly in the `D:\pose-analysis-app` folder.

---

## **How to Run the Application**

1.  **Ensure your virtual environment is activated** (your terminal prompt should start with `(venv)`).
2.  **Execute the main script from your terminal:**
    ```bash
    python main.py
    ```
3.  A webcam feed window will open, displaying the real-time pose estimation, hand tracking, angle readouts, and gesture detection.
4.  **Press the `ESC` key** to close the window and exit the application.

---

## **Understanding the AI: MediaPipe & Real-time Inference**

This project leverages the power of **Google's MediaPipe**, a cutting-edge framework for building machine learning pipelines. Instead of requiring complex model training from scratch (which demands massive datasets and computational power), MediaPipe provides highly optimized, **pre-trained deep learning models** that enable:

* **Pose Estimation:** An advanced model that accurately identifies 33 3D keypoints on the human body, even in challenging real-time scenarios.
* **Hand Tracking:** A specialized model providing precise tracking of 21 3D keypoints for each hand.

The "analysis" component of this project demonstrates practical application of these raw AI outputs:
* **Geometric Calculations:** Utilizing NumPy, the system performs real-time trigonometric calculations to derive precise angles between specific body joints (e.g., elbow, shoulder angles).
* **Heuristic-Based Gesture Recognition:** By defining logical rules and thresholds based on the relative positions and angles of various body and hand landmarks, the application intelligently interprets and recognizes common human gestures. This highlights problem-solving and the ability to build meaningful applications on top of foundational AI models.

---

## **Future Enhancements**

This project serves as a robust foundation and can be significantly expanded upon to explore more advanced applications:

* **Activity Recognition & Rep Counting:** Implement more complex logic to count repetitions for various exercises (e.g., push-ups, squats, bicep curls) based on joint angles and movement patterns.
* **Real-time Fitness Feedback:** Provide auditory or visual cues to users if their exercise form is incorrect.
* **Multi-Person Tracking:** Extend the system to track and analyze multiple individuals simultaneously.
* **Expanded Gesture Library:** Implement a wider range of custom gestures for more complex human-computer interaction or sign language interpretation.
* **Integration with Robotics/AR:** Use the precise 3D pose and hand data to control virtual avatars or interact with augmented/virtual reality environments.
* **Graphical User Interface (GUI):** Develop a more intuitive GUI (e.g., using Streamlit, Tkinter, or PyQt) for a user-friendly experience, allowing users to select activities, view performance metrics, and customize settings.

---

## **License**

This project is open-source and available under the [MIT License](LICENSE).

---

## **Contact**

* **Your Name:** Ailesh
* **GitHub:** [https://github.com/Ailesh69](https://github.com/Ailesh69)
