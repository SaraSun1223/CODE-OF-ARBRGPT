import torch

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example model
model = torch.nn.Linear(10, 10).to(device)

# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model with map_location
restored_model = torch.nn.Linear(10, 10)
restored_model.load_state_dict(torch.load('model.pth', map_location=device))