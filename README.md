# Detection Gravure

Welcome to my Detection Gravure repository! This project focuses on detection algorithms and tools for gravure printing processes. Here you'll find code, documentation, and resources to help automate and improve gravure detection workflows.

Feel free to explore!

---

## How it works

This repository provides a set of tools and scripts for:
- Generating synthetic data and low-frequency noise images ([`src/lowfreq_img.py`](src/lowfreq_img.py))
- Manipulating and augmenting normal maps ([`src/add_normals.py`](src/add_normals.py))
- Integrating normal maps to produce height maps and renderings ([`src/integrate_normals.py`](src/integrate_normals.py))
- Cropping and patch extraction from images ([`src/crop.py`](src/crop.py), [`trash/select_patch.py`](trash/select_patch.py))
- Data generation and augmentation for training deep learning models ([`src/dataGenerator.py`](src/dataGenerator.py))
- Jupyter notebooks for training and experimentation

The main workflow involves preparing datasets, augmenting them with various noise and transformations, and training detection models using the provided data generator.

---

## Installation steps

1. **Clone the repository:**
	```sh
	git clone https://github.com/steno3/detection_gravure.git
	cd detection_gravure
	```

2. **(Optional) Create virtual environment**

3. **Install dependencies:**
	```sh
	pip install -r requirements.txt
	```
    You might need other dependencies depending on your computer et python version. I used python 3.8.10 with requirements_all.txt but it might be.

4. **(Optional) Install Jupyter for running notebooks:**
	```sh
    pip install jupyter
	pip install notebook
	```

---

## How to run

1. **Ready your dataset**

    - Place your training images and ground truth masks in separate folders (no need to crop them, it is done in the data generator).
    Corresponding data should have the same names.
    The ground truth images should be black lines on white backgrounds (colors are inversed after that).
    - Update paths in the notebooks if needed.

2. **Start training**

    - Run the training script:
     ```sh
     python start_training.py <normal_images_folder> <ground_truth_folder>
     ```
    This will use the data generator ([`src/dataGenerator`](src/dataGenerator.py)) defined in the workspace to begin training.
    - You can also use the [notebook](training_with_datagen.ipynb) for step by step code.

3. **Predict with your created model**

    - After training, use the prediction script to run inference on new images:
     ```sh
     python src/predict_image.py <image_path> <model_path.h5>
     ```
    - Replace `<image_path>` with the path to your input image and `<model_path.h5>` with your trained model (`unet_model_from_normal.h5` by default)

---

## Structure

```sh
.
├── src/                   # Core source code (data generation, augmentation, integration)
│   ├── add_normals.py
│   ├── crop.py
│   ├── dataGenerator.py   # Class used to generate image batchs
│   ├── integrate_normals.py
│   ├── lowfreq_img.py
│   ├── predict_image.py   # Script to run prediction from image and model
│   └── rotate_dup.py
├── test_data_generator.py # Script to test the data generator
├── training_from_normals.ipynb
├── training_with_datagen.ipynb
├── trash/                 # Experimental or legacy scripts
│   ├── ...
│   └── select_patch.py
├── requirements.txt       # Python main dependencies
├── requirements_all.txt   # Python dependencies from my workspace (not recommended)
├── README.md
├── start_training.py      # Script to run training
└── .gitignore
```

- **src/**: Main Python modules for data processing and augmentation.
- **trash/**: Scripts for patch selection and other experiments.
- **test_data_generator.py**: Script to test the [`DataGenerator`](src/dataGenerator.py).
- **training_*.ipynb**: Jupyter notebooks for training and experimentation.

---

## References
TODO

---

## Acknowledgments

- Thanks to the open-source community for libraries such as NumPy, OpenCV, PIL, and TensorFlow/Keras.
- Special thanks to contributors and collaborators in the field of ancient engravings and computer vision.