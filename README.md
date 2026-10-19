\# Real-Time Vehicle Detection in Aerial Imagery



A computer vision project for detecting vehicles in aerial images using YOLO11n and Faster R-CNN with ResNet50 FPN.



\## Features



\- Vehicle detection in aerial images

\- YOLO11n and Faster R-CNN model comparison

\- Streamlit-based user interface

\- Evaluation using precision, recall, F1-score and mAP



\## Technology Stack



\- Python 3.12

\- PyTorch and Torchvision

\- Ultralytics YOLO

\- Streamlit, OpenCV and Pillow

\- Pandas and Matplotlib



\## Dataset



The evaluation dataset contains 102 images:

\- Training: 81 images

\- Validation: 21 images

\- Class: Car



The dataset and trained model weights are not included in this GitHub repository. Obtain the required files separately and verify their usage rights.



\## Evaluation Results



Measured on the same 21-image validation set.



| Metric | YOLO11n | Faster R-CNN |

|---|---:|---:|

| True Positives | 65 | 69 |

| False Positives | 7 | 46 |

| False Negatives | 8 | 4 |

| Precision | 90.28% | 60.00% |

| Recall | 89.04% | 94.52% |

| F1-score | 89.66% | 73.40% |

| mAP@50 | 93.71% | 95.51% |

| mAP@50–95 | 43.91% | 43.37% |

| Mean inference time | 32.63 ms/image | 62.96 ms/image |



\### Interpretation



\- YOLO11n achieved higher precision and F1-score.

\- Faster R-CNN achieved higher recall and mAP@50.

\- YOLO11n had lower measured inference time.



These results are preliminary because the validation dataset is small. Performance on other datasets may differ.



\## Setup



Open PowerShell in the project directory and run:



```powershell

py -3.12 -m venv .venv

.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip

.\\.venv\\Scripts\\python.exe -m pip install torch torchvision

.\\.venv\\Scripts\\python.exe -m pip install ultralytics streamlit pillow pandas matplotlib pyyaml opencv-python

```



\## Run the Application



After obtaining the required model weights and placing them at the paths expected by the application:



```powershell

.\\.venv\\Scripts\\python.exe -m streamlit run vehicle\_detection\_app.py

```



The model weights and dataset are excluded from this repository due to their size. The application requires the appropriate model files to run inference.



\## Limitations



\- Small validation dataset

\- Performance may vary across image conditions and datasets

\- Model weights and dataset must be obtained separately

\- The application should be tested on the target machine before deployment



\## Future Improvements



\- Evaluate on a larger dataset

\- Improve generalization

\- Add additional vehicle classes

\- Optimize inference speed



\## License



A license has not yet been specified. Verify the licensing terms of the original code and dataset before redistribution.



