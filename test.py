import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def load_and_evaluate_checkpoints(checkpoint_dir, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters (adjust these based on your original model)
    input_size = 1  # Adjust based on your input features
    hidden_size = 64  # Adjust based on your model
    num_layers = 2  # Adjust based on your model
    
    criterion = nn.MSELoss()
    
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(".pth"):
            filepath = os.path.join(checkpoint_dir, filename)
            try:
                # Initialize model
                model = LSTM(input_size, hidden_size, num_layers).to(device)
                
                # Load checkpoint
                checkpoint = torch.load(filepath)
                model.load_state_dict(checkpoint)
                
                # Evaluate model
                metrics = evaluate_model(model, test_loader, criterion, device)
                
                print(f"\nMetrics for {filename}:")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {filename}: {e}")

# Prepare your test data
def prepare_test_data(X_test, y_test, batch_size=32):
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# Example usage:
# First prepare your test data
# X_test should be your test features
# y_test should be your test targets
# test_loader = prepare_test_data(X_test, y_test)
# load_and_evaluate_checkpoints("checkpoint_lstm", test_loader)