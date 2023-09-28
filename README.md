Absolutely, let's incorporate the provided information into the README.md response:

---

# Computer Vision Puzzle Builder

This application demonstrates the construction of a puzzle using computer vision techniques. It processes input images, identifies edges and corners, and uses a Breadth-First Search (BFS) approach to assemble the puzzle pieces onto a canvas.

## Table of Contents
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Description](#description)
- [License](#license)

## Dependencies
- [natsort](https://pypi.org/project/natsort/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [scikit-image](https://scikit-image.org/)
- [imageio](https://imageio.github.io/)
- [OpenCV](https://opencv.org/)
- [tqdm](https://tqdm.github.io/)
- [cachier](https://pypi.org/project/cachier/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [networkx](https://networkx.github.io/)

Install the dependencies using `pip`:
```bash
pip install natsort numpy matplotlib seaborn scikit-image imageio opencv-python tqdm cachier scikit-learn networkx
```

## Usage
To use this application to create a puzzle:
1. Clone the repository or download the code files.
2. Install the required dependencies using the provided `pip` command.
3. Run the script and follow the instructions.

## Description
- `main.py`: Contains the main code to create the puzzle using BFS and display the results.
- `classes.py`: Defines the classes for managing puzzle pieces, edges, and the puzzle itself.

## Puzzle Creation Process
1. The script loads input images and masks, resizes them, and processes them into suitable formats.
2. Corner and edge detection are performed, identifying features necessary for the puzzle.
3. The BFS approach is employed to connect the puzzle pieces and assemble them onto a canvas.
4. The resulting puzzle is displayed, showing the canvas with the inserted pieces.

## License
This project is licensed under the [MIT License](LICENSE).

---

Feel free to further modify and add any additional details or context as needed.
