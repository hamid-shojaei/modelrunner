# ModelRunner 🚀

Welcome to `ModelRunner`! This package is designed to make your machine learning model evaluations easier and more streamlined. Whether you're running classification models, regression models, or visualizing decision trees, `ModelRunner` has got you covered.

## Features 🌟

- **Versatile Model Evaluations**: Run various ML algorithms with multiple hyperparameters and obtain predictions, scores, and feature importances.
- **Data Preprocessing**: Automatically handles label encoding for your categorical data.
- **Visualizations**: Get confusion matrix heatmaps and detailed decision tree visualizations with customizable zoom levels.

## Installation 📦

You can install `ModelRunner` directly from my GitHub repository:

```bash
pip install git+https://github.com/hamid-shojaei/modelrunner.git
```

For additional SVG functionalities, you can install the optional dependency `cairosvg`:

```bash
pip install modelrunner[svg]
```

## How to Use 🛠
Basic Model Evaluation
Initialize the ModelEvaluator with your data and desired models:

```python
from modelrunner import ModelEvaluator


hyperparameters = {
    "classification_tree": {
        "algorithm": "DecisionTree",
        "max_depth": 3
    }
}

evaluator = ModelEvaluator(X, y, hyperparameters)
evaluator.run_models()
```

Retrieve predictions, feature importances, and more!

```python
predictions = evaluator.get_predictions("classification_tree")
feature_importances = evaluator.get_feature_importances("classification_tree")
```

## Additional Installation Steps for SVG Functionality

To use SVG-related functionalities in `ModelRunner`, you'll need to install `cairosvg`. This package has its own dependencies, and depending on your system, you might need to install additional libraries.

### For Ubuntu/Debian:

```bash
sudo apt-get install libcairo2
```
### For macOS:
If you have Homebrew installed:

```bash
brew install cairo
```

After these system-level installations, you can install `cairosvg` via pip:

```bash
pip install cairosvg
```

Or simply install the SVG extras with ModelRunner:

```bash
pip install modelrunner[svg]
```

## Advanced Visualizations
Visualize your decision trees:

```python
tree_graph = evaluator.visualize_tree("classification_tree")
```

Adjust the size of your decision tree SVG visualization:

```python
from modelrunner.utils import display_svg_with_zoom_v5

display_svg_with_zoom(tree_graph, width_zoom_percent=50, height_zoom_percent=70)
```
## Author ✍️
Hamid Shojaei (hamid-shojaei)

## Last Updated
Last updated on: `August 23, 2023`

## Contributing 🤝
We welcome contributions! If you find a bug or have suggestions, please open an issue. If you're interested in contributing to the codebase, please submit a pull request.

## License ⚖️
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements 🙏
Special thanks to the open-source community and the libraries that made this project possible, including `pandas`, `scikit-learn`, `seaborn`, `matplotlib`, and `graphviz`.
