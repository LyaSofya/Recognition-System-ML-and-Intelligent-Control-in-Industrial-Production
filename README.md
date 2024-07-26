
# Intelligent Hand Operation Control System
Recognition System ML and Intelligent Control in Industrial Production

## Overview

This project is aimed at developing an intelligent system for controlling manual operations in industrial production using machine learning (ML) techniques. The system enhances production quality and integrates new assembly scenarios. The testing setup includes an automated workstation with a computer, monitor, projector, and Basler industrial cameras. It monitors operations in real-time, identifying typical errors and safety violations to reduce defects, improve productivity, and decrease accidents.

## Repository Structure

The repository is organized into the following directories:

- **code/**: Contains the source code for the project.
- **doc/**: Contains all project-related documents:
  - `Developer Documentation.docx`
  - `Developer Documentation.pdf`
  - `Full report.docx`
  - `Full report.pdf`
  - `Performance Improvement Research Report.docx`
  - `Performance Improvement Research Report.pdf`
  - `Presentation.pdf`
  - `Presentation.pptx`
  - `User Documentation.docx`
  - `User Documentation.pdf`
  - `terms of reference (TOR).pdf`
- **Video.mp4**: A video demonstrating the system in action.

## Machine Learning Details

### Technology Stack

- **Programming Language**: Python
- **Deep Learning Frameworks**: 
  - YOLO v8 for object detection
  - MediaPipe for hand tracking and gesture recognition
- **Libraries and Tools**:
  - OpenCV for computer vision tasks
  - TensorFlow and PyTorch for model training and inference
  - NumPy and Pandas for data manipulation and analysis

### Key Features

- **Object Tracking Algorithms**: Implemented to enhance detection quality, including classic tracking algorithms, correlation filters, and deep tracking methods.
- **Real-time Performance**: Achieved through parallel and concurrent computations leveraging CPU and GPU resources.
- **Data Collection and Annotation**: Extensive data gathering and labeling to improve detection accuracy, focusing on small parts and complex assembly processes.

### Experiments and Implementation

1. **Object Tracking**:
   - Evaluated various tracking algorithms, including classic and deep learning-based methods.
   - Developed a module for integrating tracking with detection algorithms to enhance stability and accuracy.

2. **Detection Models**:
   - Addressed common issues such as detection misses, instability, and false positives.
   - Implemented and tested YOLO v8 models, improving detection reliability.

3. **Application Scenarios**:
   - Developed specific scenarios for assembly tasks, such as moving screws and assembling drone motors.
   - Enhanced error correction and component tracking throughout the assembly process.

4. **Performance Optimization**:
   - Conducted research on parallel and concurrent computation methods.
   - Improved FPS by offloading computations to GPU and implementing multithreading and multiprocessing techniques.

### Results

The system was successfully tested on the laboratory setup and is ready for deployment in real-world industrial environments. It demonstrated high potential for commercial use, significantly reducing errors and improving operational efficiency.

## Documentation

Detailed documentation is provided in the `doc/` folder, covering both developer and user aspects:

- **Developer Documentation**: Guides for developers to understand the system's architecture and codebase.
- **User Documentation**: Instructions for end-users to operate the system effectively.
- **Full Report**: Comprehensive report on the project's development, experiments, and results.
- **Performance Improvement Research Report**: In-depth research on methods and results for optimizing system performance.
- **Presentation**: Summary of the project, suitable for stakeholders and presentations.

## Demonstration

A video demonstration of the system in action is available in `Video.mp4`.
