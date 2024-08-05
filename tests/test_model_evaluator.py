import pandas as pd
from mypackage.model_evaluator import ModelEvaluator

def test_model_evaluation():
    # Sample data
    data = {
        'feature1': ['A', 'B', 'A', 'C'],
        'feature2': [1, 2, 3, 4],
        'target': [1, 0, 1, 0]
    }

    df = pd.DataFrame(data)
    X = df[['feature1', 'feature2']]
    y = df['target']

    hyperparameters = {
        "classification_tree": {
            "algorithm": "DecisionTree",
            "max_depth": 3
        }
    }

    evaluator = ModelEvaluator(X, y, hyperparameters)
    evaluator.run_models()
    
    # Test predictions
    predictions = evaluator.get_predictions("classification_tree")
    assert predictions is not None
    assert len(predictions) == len(y)

    # More tests can be added here for other methods and functionalities for this

