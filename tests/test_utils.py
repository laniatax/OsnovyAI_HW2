import unittest
import torch
from homework_model_modification import LinearRegressionWithRegularization, LogisticRegressionMulticlass
from homework_datasets import CustomCSVDataset
import os

class TestModels(unittest.TestCase):

    def test_linear_reg_regularization_loss(self):
        model = LinearRegressionWithRegularization(input_dim=3)
        loss = model.regularization_loss()
        self.assertIsInstance(loss, torch.Tensor)

    def test_logistic_predict_shape(self):
        model = LogisticRegressionMulticlass(input_dim=4, num_classes=3)
        x = torch.randn(5, 4)
        preds = model.predict(x)
        self.assertEqual(preds.shape, torch.Size([5]))

class TestDataset(unittest.TestCase):

    def test_dataset_loading(self):
        # Используйте реальный небольшой csv файл или сгенерируйте тестовый DataFrame
        import pandas as pd
        test_csv = "tests/test_data.csv"
        df = pd.DataFrame({
            'feature1': [1, 2],
            'feature2': [3, 4],
            'target': [0, 1]
        })
        df.to_csv(test_csv, index=False)

        dataset = CustomCSVDataset(test_csv, target_column='target')
        self.assertEqual(len(dataset), 2)
        x, y = dataset[0]
        self.assertEqual(len(x), 2)  # 2 признака
        os.remove(test_csv)

if __name__ == '__main__':
    unittest.main()
