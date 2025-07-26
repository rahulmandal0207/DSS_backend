# Driver Surveillance System (DSS)

The **Driver Surveillance System (DSS)** is a real-time monitoring solution designed to enhance road safety by detecting driver drowsiness and inattentiveness using computer vision and machine learning techniques.

## 🚀 Key Features

- **Eye Closure Detection**  
  Monitors the driver’s eye aspect ratio (EAR) to detect prolonged eye closure, a primary indicator of drowsiness.

- **Yawning Detection**  
  Detects mouth opening using facial landmarks to identify yawning behavior, which often precedes sleepiness.

- **Head Tilt Detection**  
  Analyzes the orientation of the driver's head to determine inattention or drowsiness based on unusual tilt patterns.

## 🧠 Technologies Used

- **Python** – Core programming language for backend logic and integration.
- **OpenCV** – For real-time video capture and image processing.
- **MediaPipe** – For efficient facial landmark detection.
- **Machine Learning Libraries** – Used for behavioral pattern analysis and threshold-based decision making.



## For the collaborators

Step 1: Clone the repo

```commandline
git clone https://github.com/rahulmandal0207/DSS_backend.git
```

Step 2: Create a new branch __[ important ]__

```commandline
git checkout -b branch_name
```

Step 3: Make a new virtual environment

```commandline
python -m venv env_name
```

Step 4: Install all the requirements

```commandline
pip install -r requirements.txt
```

Step 5: Create a new file for start working

Step 6: Do not forget to add the virtual_environment/resource_folder to the .gitignore file
