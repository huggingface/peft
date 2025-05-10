import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision import datasets, transforms
import copy

def calculate_adapter_ranks(model, threshold=1.0):
    adapter_ranks = []
    
    # Iterate through all layers except the last one (output layer)
    for i in range(len(model.layers) - 1):
        # Get the weight matrix for this layer
        layer_matrix = model.layers[i].weight.detach().clone()
        
        # Calculate singular values
        singular_values = torch.svd(layer_matrix).S
        
        # Count singular values below threshold
        rank = (singular_values < threshold).cpu().numpy().sum()
        
        # Add to our list of ranks
        adapter_ranks.append(rank)
        
        # Optional: log the rank for this layer
        print(f"Layer {i}: {rank} singular values below {threshold}")
    
    return adapter_ranks

def plot_singular_values_torch(matrix):
    # Compute singular values using SVD in torch
    U, S, V = torch.svd(matrix)
    
    # Convert singular values tensor to a list for plotting
    singular_values = S.tolist()
    
    # Plot the singular values
    plt.figure(figsize=(16, 6))
    plt.plot(singular_values, 'o-', label='Singular Values')
    plt.title('Singular Values of the Matrix (Torch)')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.grid(True)
    plt.legend()
    plt.show()



def count_parameters(model: nn.Module, model_name:str = "Model"):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model {model_name} Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    return total_params, trainable_params


def train_model(model, x, y, validation_split=0.1, batch_size=512, epoch_num=15, lr=1e-4, weight_decay=1e-6, patience=2):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

    # Split the data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split, random_state=42)

    # Training data loader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Validation data loader
    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    # Compute and print initial training and validation loss before any training
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for this part
        # Initial training loss
        outputs = model(x_train)
        initial_train_loss = criterion(outputs, y_train)
        
        # Initial validation loss
        val_outputs = model(x_val)
        initial_val_loss = criterion(val_outputs, y_val)
        
        print(f'Initial Loss - Training: {initial_train_loss.item():.8f}, Validation: {initial_val_loss.item():.8f}')

    # Training loop
    for epoch in range(epoch_num):
        model.train()  # Set model back to training mode
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            # Forward pass: get predictions from the model
            outputs = model(batch_x)

            # Compute the loss
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Compute training loss for the epoch
        train_loss = running_loss / len(train_loader)

        # Validation loss at the end of each epoch
        model.eval()  # Set model to evaluation mode for validation
        with torch.no_grad():  # Disable gradient calculation
            val_loss = 0.0
            for val_batch_x, val_batch_y in val_loader:
                val_outputs = model(val_batch_x)
                val_loss += criterion(val_outputs, val_batch_y).item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.8f}")
            break

        # Print training and validation loss for the epoch
        print(f'Epoch [{epoch+1}/{epoch_num}], Training Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}')

    print("Training complete.")


def compute_class_weights(train_loader, balancing_method, num_classes=None, rare_class_weight=0.5, device='cpu'):
    weights = None

    # Extract labels from the dataset
    if hasattr(train_loader.dataset, 'tensors'):
        all_labels = train_loader.dataset.tensors[1]
    else:
        all_labels = torch.cat([batch[1] for batch in train_loader], dim=0)
    
    # Ensure labels are on CPU and flattened to 1D
    all_labels = all_labels.cpu().view(-1)
    
    # Determine number of classes if not provided
    if num_classes is None:
        num_classes = int(all_labels.max().item()) + 1
    
    # Count occurrences for each class
    counts = torch.bincount(all_labels, minlength=num_classes).float()

    if balancing_method == "inverse_frequency":
        total_count = all_labels.numel()  # Total number of samples
        # Compute weights as total_samples / samples_per_class
        weights = total_count / counts
    
    # Identify the class with the highest computed weight (i.e. the rarest class)
    rare_class_index = torch.argmax(weights)

    # Scale the rare class weight by rare_class_weight and scale all others by (1 - rare_class_weight)
    adjusted_weights = weights.clone() * (1 - rare_class_weight)
    adjusted_weights[rare_class_index] = weights[rare_class_index] * rare_class_weight

    # Return the adjusted weights on the specified device
    return adjusted_weights.to(device)




def train_model_classification(
    model,
    train_loader,
    val_loader,
    epoch_num=15,
    lr=5e-4,
    weight_decay=1e-6,
    patience=10,
    to_print=True,
    logger=None,
    model_name="Model",
    weights=None
):
    if weights is not None:
        criterion_train = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion_train = nn.CrossEntropyLoss()
    criterion_eval = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, threshold=1e-5
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epoch_num):
        model.train()
        running_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(next(model.parameters()).device)
            batch_y = batch_y.to(next(model.parameters()).device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion_train(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch_x, val_batch_y in val_loader:
                val_batch_x = val_batch_x.to(next(model.parameters()).device)
                val_batch_y = val_batch_y.to(next(model.parameters()).device)
                val_outputs = model(val_batch_x)
                val_loss += criterion_eval(val_outputs, val_batch_y).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if to_print:
                print(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.8f}")
            break

        if to_print:
            print(
                f"[{model_name}] Epoch [{epoch+1}/{epoch_num}], "
                f"Training Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}, LR: {current_lr:.8f}"
            )

        if logger is not None and epoch % 10 == 0:
            logger.report_scalar(title=f"{model_name} Loss", series="Training", value=train_loss, iteration=epoch)
            logger.report_scalar(title=f"{model_name} Loss", series="Validation", value=val_loss, iteration=epoch)
            logger.report_scalar(title=f"{model_name} LR", series="Learning Rate", value=current_lr, iteration=epoch)

    if to_print:
        print(f"[{model_name}] Training complete.")

    return best_val_loss


def evaluate_model(model, x_val, y_val, batch_size=2048, to_print=True):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    val_dataset = TensorDataset(x_val, y_val)
    criterion = nn.MSELoss()

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle validation data
    total_samples = len(val_dataset)

    with torch.no_grad():  # Disable gradient calculation for validation
        for val_x, val_y in val_loader:
            # val_y = val_y.view(-1, 1)  # Reshape target if necessary

            output = model(val_x)
            batch_loss = criterion(output, val_y)
            total_loss += batch_loss.item() * val_x.size(0)  # Multiply by batch size to accumulate total loss

    # Average the losses over all validation samples
    average_loss = total_loss / total_samples

    if to_print:
        print(f'Model Validation Loss: {average_loss:.8f}')

    return average_loss


def evaluate_model_classification(model, val_loader, to_print=True):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()  # For classification tasks

    with torch.no_grad():  # Disable gradient calculation for validation
        for val_x, val_y in val_loader:
            val_x, val_y = val_x.to(next(model.parameters()).device), val_y.to(next(model.parameters()).device)
            
            # Forward pass
            output = model(val_x)
            
            # Compute loss
            batch_loss = criterion(output, val_y)
            
            # Accumulate total loss
            total_loss += batch_loss.item() * val_x.size(0)
            total_samples += val_x.size(0)

    # Compute the average loss
    average_loss = total_loss / total_samples

    if to_print:
        print(f'Model Validation Loss: {average_loss:.8f}')

    return average_loss


def calc_performance_per_scaling_factor(model, max_point, step, X_pretrain_test, y_pretrain_test, X_finetune_test, y_finetune_test, X_combined_test, y_combined_test):
    pretrain_performance = []
    finetune_performance = []
    combined_performance = []
    scaling_factors_range = range(0, max_point, step)
    for new_scaling_factor in scaling_factors_range:
        model.scaling_factor = new_scaling_factor
        pretrain_performance.append(evaluate_model(model, X_pretrain_test, y_pretrain_test, to_print=False))
        finetune_performance.append(evaluate_model(model, X_finetune_test, y_finetune_test, to_print=False))
        combined_performance.append(evaluate_model(model, X_combined_test, y_combined_test, to_print=False))

    model.scaling_factor = 1
    
    return pretrain_performance, finetune_performance, combined_performance, scaling_factors_range

def calc_performance_per_scaling_factor2(model, max_point, step, pretrain_test_loader, finetune_test_loader, combined_test_loader):
    pretrain_performance = []
    finetune_performance = []
    combined_performance = []
    scaling_factors_range = np.arange(0, max_point + step, step).tolist()

    for new_scaling_factor in scaling_factors_range:
        model.scaling_factor = new_scaling_factor  # Adjust scaling factor

        # Evaluate performance using loaders
        pretrain_performance.append(evaluate_model_classification(model, pretrain_test_loader, to_print=False))
        finetune_performance.append(evaluate_model_classification(model, finetune_test_loader, to_print=False))
        combined_performance.append(evaluate_model_classification(model, combined_test_loader, to_print=False))

    # Reset scaling factor to default
    model.scaling_factor = 1

    return pretrain_performance, finetune_performance, combined_performance, scaling_factors_range

def get_fashion_mnist_datasets(device, fine_tune_classes=[9], drizzling=0):
    # 1) Load FashionMNIST train and test sets
    fashion_mnist_train = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transforms.ToTensor()
    )
    fashion_mnist_test = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transforms.ToTensor()
    )

    # 2) Convert train and test data to tensors and normalize to [0,1]
    X_train = fashion_mnist_train.data.unsqueeze(1).float() / 255.0
    y_train = fashion_mnist_train.targets
    X_test = fashion_mnist_test.data.unsqueeze(1).float() / 255.0
    y_test = fashion_mnist_test.targets

    # 3) For the training set:
    # a) Get pre-training samples for labels NOT in fine_tune_classes
    mask_pretrain_train = torch.ones_like(y_train, dtype=torch.bool)
    for cls in fine_tune_classes:
        mask_pretrain_train &= (y_train != cls)
    X_pretrain_train = X_train[mask_pretrain_train]
    y_pretrain_train = y_train[mask_pretrain_train]

    # b) For each fine-tuning class:
    #    - Select the first 'drizzling' samples to add to the pre-training set.
    #    - Use the remaining samples for fine-tuning.
    pretrain_drizzle_list = []
    pretrain_drizzle_labels = []
    finetune_train_list = []
    finetune_train_labels = []
    
    for cls in fine_tune_classes:
        indices = (y_train == cls).nonzero(as_tuple=True)[0]
        # Get the first drizzling samples from class 'cls' (if drizzling > 0)
        if drizzling > 0:
            drizzle_indices = indices[:drizzling]
            finetune_indices = indices[drizzling:]
        else:
            drizzle_indices = indices[:0]  # empty
            finetune_indices = indices
        pretrain_drizzle_list.append(X_train[drizzle_indices])
        pretrain_drizzle_labels.append(y_train[drizzle_indices])
        finetune_train_list.append(X_train[finetune_indices])
        finetune_train_labels.append(y_train[finetune_indices])
    
    # Append drizzled fine-tuning samples to pre-training set
    if pretrain_drizzle_list:
        X_pretrain_train = torch.cat([X_pretrain_train, *pretrain_drizzle_list], dim=0)
        y_pretrain_train = torch.cat([y_pretrain_train, *pretrain_drizzle_labels], dim=0)
    
    # Concatenate fine-tuning samples for training
    X_finetune_train = torch.cat(finetune_train_list, dim=0)
    y_finetune_train = torch.cat(finetune_train_labels, dim=0)
    
    # Move training datasets to device
    X_pretrain_train = X_pretrain_train.to(device)
    y_pretrain_train = y_pretrain_train.to(device)
    X_finetune_train = X_finetune_train.to(device)
    y_finetune_train = y_finetune_train.to(device)

    # Combine pretrain and finetune training sets for the overall training loader
    X_pre_ft_train = torch.cat([X_pretrain_train, X_finetune_train], dim=0)
    y_pre_ft_train = torch.cat([y_pretrain_train, y_finetune_train], dim=0)

    # 4) Process the test set
    # First, split the test set into two portions (80%/20%)
    X_test_combined_pre_fine, X_test_combined, y_test_combined_pre_fine, y_test_combined = train_test_split(
        X_test, y_test, test_size=0.2, random_state=42, stratify=y_test
    )
    
    # Move the split test portion to device
    X_test_combined_pre_fine = X_test_combined_pre_fine.to(device)
    y_test_combined_pre_fine = y_test_combined_pre_fine.to(device)
    
    # a) Get pre-training test samples: labels not in fine_tune_classes
    mask_pretrain_test = torch.ones_like(y_test_combined_pre_fine, dtype=torch.bool)
    for cls in fine_tune_classes:
        mask_pretrain_test &= (y_test_combined_pre_fine != cls)
    X_pretrain_test = X_test_combined_pre_fine[mask_pretrain_test]
    y_pretrain_test = y_test_combined_pre_fine[mask_pretrain_test]
    
    # b) For each fine-tuning class on the test set:
    pretrain_drizzle_list_test = []
    pretrain_drizzle_labels_test = []
    finetune_test_list = []
    finetune_test_labels = []
    
    for cls in fine_tune_classes:
        indices = (y_test_combined_pre_fine == cls).nonzero(as_tuple=True)[0]
        # Get the drizzling samples (first 'drizzling' samples) and the remaining fine-tuning samples
        if drizzling > 0:
            drizzle_indices = indices[:drizzling]
            finetune_indices = indices[drizzling:]
        else:
            drizzle_indices = indices[:0]
            finetune_indices = indices
        pretrain_drizzle_list_test.append(X_test_combined_pre_fine[drizzle_indices])
        pretrain_drizzle_labels_test.append(y_test_combined_pre_fine[drizzle_indices])
        finetune_test_list.append(X_test_combined_pre_fine[finetune_indices])
        finetune_test_labels.append(y_test_combined_pre_fine[finetune_indices])
    
    # Add drizzled samples to pretrain test set
    if pretrain_drizzle_list_test:
        X_pretrain_test = torch.cat([X_pretrain_test, *pretrain_drizzle_list_test], dim=0)
        y_pretrain_test = torch.cat([y_pretrain_test, *pretrain_drizzle_labels_test], dim=0)
    
    # Concatenate fine-tuning samples for test set
    X_finetune_test = torch.cat(finetune_test_list, dim=0).to(device)
    y_finetune_test = torch.cat(finetune_test_labels, dim=0).to(device)
    
    # Combine pretrain and fine-tune test sets
    X_pre_ft_test = torch.cat([X_pretrain_test, X_finetune_test], dim=0).to(device)
    y_pre_ft_test = torch.cat([y_pretrain_test, y_finetune_test], dim=0).to(device)
    
    # The combined test set (80%/20% split remainder) remains unchanged
    X_combined_test = X_test_combined.to(device)
    y_combined_test = y_test_combined.to(device)

    # 5) Flatten the data for the linear layers (FashionMNIST images are 28x28)
    X_pretrain_train = X_pretrain_train.view(X_pretrain_train.size(0), -1)
    X_finetune_train = X_finetune_train.view(X_finetune_train.size(0), -1)
    X_pretrain_test = X_pretrain_test.view(X_pretrain_test.size(0), -1)
    X_finetune_test = X_finetune_test.view(X_finetune_test.size(0), -1)
    X_combined_test = X_combined_test.view(X_combined_test.size(0), -1)
    X_pre_ft_train = X_pre_ft_train.view(X_pre_ft_train.size(0), -1)
    X_pre_ft_test = X_pre_ft_test.view(X_pre_ft_test.size(0), -1)

    # 6) Create data loaders
    batch_size = 512
    pretrain_train_loader = DataLoader(
        TensorDataset(X_pretrain_train, y_pretrain_train), 
        batch_size=batch_size, 
        shuffle=True
    )
    finetune_train_loader = DataLoader(
        TensorDataset(X_finetune_train, y_finetune_train), 
        batch_size=batch_size, 
        shuffle=True
    )
    pretrain_test_loader = DataLoader(
        TensorDataset(X_pretrain_test, y_pretrain_test), 
        batch_size=batch_size, 
        shuffle=False
    )
    finetune_test_loader = DataLoader(
        TensorDataset(X_finetune_test, y_finetune_test), 
        batch_size=batch_size, 
        shuffle=False
    )
    combined_test_loader = DataLoader(
        TensorDataset(X_combined_test, y_combined_test), 
        batch_size=batch_size, 
        shuffle=False
    )
    pre_ft_train_loader = DataLoader(
        TensorDataset(X_pre_ft_train, y_pre_ft_train), 
        batch_size=batch_size, 
        shuffle=True
    )
    pre_ft_test_loader = DataLoader(
        TensorDataset(X_pre_ft_test, y_pre_ft_test), 
        batch_size=batch_size, 
        shuffle=False
    )

    # 7) Return the data loaders
    return (
        pretrain_train_loader, 
        finetune_train_loader, 
        pretrain_test_loader, 
        finetune_test_loader, 
        combined_test_loader, 
        pre_ft_train_loader, 
        pre_ft_test_loader
    )




def get_mnist_datasets(device):
    mnist_train = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())
    # Convert train and test data to tensors
    X_train = mnist_train.data.unsqueeze(1).float() / 255.0  # Normalize to [0, 1]
    y_train = mnist_train.targets
    X_test = mnist_test.data.unsqueeze(1).float() / 255.0
    y_test = mnist_test.targets

    # Split pretrain (0-8) and finetune (9) data
    X_pretrain_train = X_train[y_train < 9].to(device)
    y_pretrain_train = y_train[y_train < 9].to(device)

    X_finetune_train = X_train[y_train == 9].to(device)
    y_finetune_train = y_train[y_train == 9].to(device)

    # Combine pretrain and finetune datasets for training the model that was trained on everything
    X_pre_ft_train = torch.cat([X_pretrain_train, X_finetune_train], dim=0)
    y_pre_ft_train = torch.cat([y_pretrain_train, y_finetune_train], dim=0)

    X_test_combined_pre_fine, X_test_combined, y_test_combined_pre_fine, y_test_combined = train_test_split(
        X_test, y_test, test_size=0.2, random_state=42, stratify=y_test
    )

    X_pretrain_test = X_test_combined_pre_fine[y_test_combined_pre_fine < 9].to(device)
    y_pretrain_test = y_test_combined_pre_fine[y_test_combined_pre_fine < 9].to(device)

    X_finetune_test = X_test_combined_pre_fine[y_test_combined_pre_fine == 9].to(device)
    y_finetune_test = y_test_combined_pre_fine[y_test_combined_pre_fine == 9].to(device)

    X_pre_ft_test = torch.cat([X_pretrain_test, X_finetune_test], dim=0)
    y_pre_ft_test = torch.cat([y_pretrain_test, y_finetune_test], dim=0)

    # Combined test set (all test samples)
    X_combined_test = X_test_combined.to(device)
    y_combined_test = y_test_combined.to(device)

    # Flatten the data for linear layers (MNIST images are 28x28, so flatten to 784)
    X_pretrain_train = X_pretrain_train.view(X_pretrain_train.size(0), -1)
    X_finetune_train = X_finetune_train.view(X_finetune_train.size(0), -1)
    X_pretrain_test = X_pretrain_test.view(X_pretrain_test.size(0), -1)
    X_finetune_test = X_finetune_test.view(X_finetune_test.size(0), -1)
    X_combined_test = X_combined_test.view(X_combined_test.size(0), -1)
    X_pre_ft_train = X_pre_ft_train.view(X_pre_ft_train.size(0), -1)
    X_pre_ft_test = X_pre_ft_test.view(X_pre_ft_test.size(0), -1)

    # Create data loaders
    batch_size = 512
    pretrain_train_loader = DataLoader(TensorDataset(X_pretrain_train, y_pretrain_train), batch_size=batch_size, shuffle=True)
    finetune_train_loader = DataLoader(TensorDataset(X_finetune_train, y_finetune_train), batch_size=batch_size, shuffle=True)
    pretrain_test_loader = DataLoader(TensorDataset(X_pretrain_test, y_pretrain_test), batch_size=batch_size, shuffle=False)
    finetune_test_loader = DataLoader(TensorDataset(X_finetune_test, y_finetune_test), batch_size=batch_size, shuffle=False)
    combined_test_loader = DataLoader(TensorDataset(X_combined_test, y_combined_test), batch_size=batch_size, shuffle=False)
    pre_ft_train_loader = DataLoader(TensorDataset(X_pre_ft_train, y_pre_ft_train), batch_size=batch_size, shuffle=True)
    pre_ft_test_loader = DataLoader(TensorDataset(X_pre_ft_test, y_pre_ft_test), batch_size=batch_size, shuffle=True)

    return pretrain_train_loader, finetune_train_loader, pretrain_test_loader, finetune_test_loader, combined_test_loader, pre_ft_train_loader, pre_ft_test_loader


# Define the generalized base model
class BaseModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(BaseModel, self).__init__()
        self.layers = nn.ModuleList()  # Store layers in a list

        # Add first hidden layer
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size, bias=False))
            prev_size = hidden_size

        # Add the output layer
        self.layers.append(nn.Linear(prev_size, output_size, bias=False))

    def forward(self, x):
        # Pass through hidden layers with ReLU activation
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        
        # Pass through the output layer
        x = self.layers[-1](x)
        return x
    
class GeneralAdapterModel(nn.Module):
    def __init__(self, base_model, adapters_dict, classification_matrix, scaling_factor=1.0, activation=nn.ReLU()):
        super(GeneralAdapterModel, self).__init__()
        self.scaling_factor = scaling_factor
        self.activation = activation
        self.base_model = copy.deepcopy(base_model)
        self.classification_matrix = classification_matrix

        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        # adapters_dict: {layer_idx: adapter_matrix}
        self.adapters_dict = adapters_dict

    def forward(self, x):
        hidden_state = x
        num_layers = len(self.base_model.layers)

        for idx in range(num_layers - 1):  # excluding the final output layer
            base_weight = self.base_model.layers[idx].weight
            bias = self.base_model.layers[idx].bias

            # Add adapter if exists for this layer
            if idx in self.adapters_dict:
                adapted_weight = base_weight + self.scaling_factor * self.adapters_dict[idx]
            else:
                adapted_weight = base_weight

            hidden_state = torch.matmul(hidden_state, adapted_weight.T)
            if bias is not None:
                hidden_state += bias

            hidden_state = self.activation(hidden_state)

        # Final output layer (unchanged)
        # return self.base_model.layers[-1](hidden_state)
        return hidden_state @ self.classification_matrix.T

class FullFtAdapter(nn.Module):
    def __init__(self, input_size, hidden_sizes, base_model, scaling_factor=1, ft_layers=[0], activation=nn.ReLU()):
        super(FullFtAdapter, self).__init__()
        self.scaling_factor = scaling_factor
        self.ft_layers = ft_layers
        self.activation = activation
        self.base_model = copy.deepcopy(base_model)

        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        self.full_adapter = nn.ModuleList()

        prev_size = input_size

        for idx, hidden_size in enumerate(hidden_sizes):
            if idx in ft_layers:
                new_adapter = nn.Linear(prev_size, hidden_size, bias=False)
                torch.nn.init.kaiming_uniform_(new_adapter.weight)
                self.full_adapter.append(new_adapter)
            else:
                self.full_adapter.append(None)
            prev_size = hidden_size

    def forward(self, x):
        hidden_state = x
        for idx, full_adapter in enumerate(self.full_adapter):
            if idx in self.ft_layers:
                w_combined = self.calc_new_w(idx)
                hidden_state = torch.matmul(hidden_state, w_combined.T)
                layer_bias = self.base_model.layers[idx].bias
                if layer_bias is not None:
                    hidden_state += layer_bias
            else:
                # Use the base model's layer directly
                hidden_state = self.base_model.layers[idx](hidden_state)

            # Apply activation (ReLU, or as specified)
            hidden_state = self.activation(hidden_state)

        # Pass through the final layer (output layer)
        return self.base_model.layers[-1](hidden_state)

    def calc_new_w(self, layer_index):
        """
        Calculate combined weights (base weights + LoRA weights) for the specified layer.
        """
        base_weights = self.base_model.layers[layer_index].weight
        return base_weights + self.scaling_factor * self.full_adapter[layer_index].weight
    

class LoRAAdapter(nn.Module):
    def __init__(self, input_size, hidden_sizes, base_model, ranks=None, scaling_factor=1, lora_layer_indices=[0], activation=nn.ReLU()):
        super(LoRAAdapter, self).__init__()
        self.scaling_factor = scaling_factor
        self.lora_layer_indices = lora_layer_indices
        self.activation = activation
        self.base_model = copy.deepcopy(base_model)

        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        self.lora_A = nn.ModuleList()
        self.lora_B = nn.ModuleList()

        prev_size = input_size

        for idx, (hidden_size, rank) in enumerate(zip(hidden_sizes, ranks)):
            if idx in lora_layer_indices:
                self.lora_A.append(nn.Linear(prev_size, rank, bias=False))
                self.lora_B.append(nn.Linear(rank, hidden_size, bias=False))
                nn.init.zeros_(self.lora_B[-1].weight)
                nn.init.normal_(self.lora_A[-1].weight, mean=0.0, std=0.1)
            else:
                self.lora_A.append(None)
                self.lora_B.append(None)
            prev_size = hidden_size

    def forward(self, x):
        hidden_state = x
        for idx, (lora_A, lora_B) in enumerate(zip(self.lora_A, self.lora_B)):
            if idx in self.lora_layer_indices:
                # Apply LoRA weights to the selected layer
                w_combined = self.calc_new_w(idx)
                hidden_state = torch.matmul(hidden_state, w_combined.T)
                layer_bias = self.base_model.layers[idx].bias
                if layer_bias is not None:
                    hidden_state += layer_bias
            else:
                # Use the base model's layer directly
                hidden_state = self.base_model.layers[idx](hidden_state)

            # Apply activation (ReLU, or as specified)
            hidden_state = self.activation(hidden_state)

        # Pass through the final layer (output layer)
        return self.base_model.layers[-1](hidden_state)

    def calc_new_w(self, layer_index):
        """
        Calculate combined weights (base weights + LoRA weights) for the specified layer.
        """
        BA_weights = torch.matmul(self.lora_B[layer_index].weight, self.lora_A[layer_index].weight)
        base_weights = self.base_model.layers[layer_index].weight
        return base_weights + self.scaling_factor * BA_weights
    

class LinLoRAAdapter(nn.Module):
    def __init__(self, base_model, ranks=4, scaling_factor=1, linlora_layer_indices=[0], activation=nn.ReLU(), enforce_sv_positive=True):
        super(LinLoRAAdapter, self).__init__()
        self.base_model = copy.deepcopy(base_model)
        self.scaling_factor = scaling_factor
        self.linlora_layer_indices = linlora_layer_indices  # Renamed for consistency
        self.activation = activation
        self.enforce_sv_positive = enforce_sv_positive

        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.eval()  # Keep the base model in evaluation mode

        # Prepare to store adapters for each layer
        self.adapters = nn.ParameterList()
        self.U_directions = []
        self.V_directions = []

        # Collect layers from the base model
        self.base_layers = self.base_model.layers  # Assuming base_model.layers is nn.ModuleList

        # Ensure ranks is a list matching the number of layers
        if isinstance(ranks, int):
            ranks = [ranks] * len(self.base_layers)

        # Initialize adapters and singular vectors for specified layers
        for idx, (layer, rank) in enumerate(zip(self.base_layers[:-1], ranks)):  # Exclude the output layer
            if idx in self.linlora_layer_indices:  # Updated variable name
                weight = layer.weight.data
                # Compute SVD
                U_r, V_r = self.get_matrix_U_and_V(weight, rank)
                self.U_directions.append(U_r)
                self.V_directions.append(V_r)
                # Initialize adapter parameters
                adapter = nn.Parameter(torch.ones(rank) * 1e-7)
                self.adapters.append(adapter)
            else:
                # Placeholders for layers without DLinLoRA
                self.U_directions.append(None)
                self.V_directions.append(None)
                self.adapters.append(None)

    def forward(self, x):
        hidden_state = x
        for idx, layer in enumerate(self.base_layers[:-1]):  # Exclude the output layer
            if idx in self.linlora_layer_indices:  # Updated variable name
                # Calculate the modified weights
                w_combined = self.calc_new_w(idx)
                hidden_state = torch.matmul(hidden_state, w_combined.T)
                if layer.bias is not None:
                    hidden_state += layer.bias
            else:
                # Use the base layer directly
                hidden_state = layer(hidden_state)
            # Apply activation
            hidden_state = self.activation(hidden_state)
        # Pass through the output layer
        output = self.base_layers[-1](hidden_state)
        return output

    def calc_new_w(self, idx):
        if self.enforce_sv_positive:
            adapter = torch.diag(torch.relu(self.adapters[idx]))
        else:
            adapter = torch.diag(self.adapters[idx])
            
        # Compute the adapter
        adapter_matrix = self.U_directions[idx] @ adapter @ self.V_directions[idx]
        # Combine with base weights
        base_weights = self.base_layers[idx].weight
        return base_weights + self.scaling_factor * adapter_matrix

    def get_matrix_U_and_V(self, weight_matrix, rank):
        # Compute SVD
        U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
        # Get indices of smallest singular values
        sorted_indices = torch.argsort(S)
        smallest_r_indices = sorted_indices[:rank]
        # Extract corresponding singular vectors
        U_r = U[:, smallest_r_indices]
        V_r = Vh[smallest_r_indices, :]
        return U_r, V_r
    

class DLinLoRAAdapter(nn.Module):
    def __init__(self, base_model, ranks=4, scaling_factor=1, linlora_layer_indices=[0], activation=nn.ReLU(), enforce_sv_positive=True):
        super(DLinLoRAAdapter, self).__init__()
        self.base_model = copy.deepcopy(base_model)
        self.scaling_factor = scaling_factor
        self.linlora_layer_indices = linlora_layer_indices  # Renamed for consistency
        self.activation = activation
        self.enforce_sv_positive = enforce_sv_positive

        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.eval()  # Keep the base model in evaluation mode

        # Prepare to store adapters for each layer
        self.adapters = nn.ParameterList()
        self.scaling_adapters = nn.ParameterList()
        self.U_directions = []
        self.V_directions = []

        # Collect layers from the base model
        self.base_layers = self.base_model.layers  # Assuming base_model.layers is nn.ModuleList

        # Ensure ranks is a list matching the number of layers
        if isinstance(ranks, int):
            ranks = [ranks] * len(self.base_layers)

        # Initialize adapters and singular vectors for specified layers
        for idx, (layer, rank) in enumerate(zip(self.base_layers[:-1], ranks)):  # Exclude the output layer
            if idx in self.linlora_layer_indices:  # Updated variable name
                weight = layer.weight.data
                # Compute SVD
                U_r, V_r = self.get_matrix_U_and_V(weight, rank)
                self.U_directions.append(U_r)
                self.V_directions.append(V_r)
                # Initialize adapter parameters
                adapter = nn.Parameter(torch.ones(rank) * 1e-7)
                scaling_adapter = nn.Parameter(torch.ones(weight.shape[1]))
                self.adapters.append(adapter)
                self.scaling_adapters.append(scaling_adapter)
            else:
                # Placeholders for layers without DLinLoRA
                self.U_directions.append(None)
                self.V_directions.append(None)
                self.adapters.append(None)
                self.scaling_adapters.append(None)

    def forward(self, x):
        hidden_state = x
        for idx, layer in enumerate(self.base_layers[:-1]):  # Exclude the output layer
            if idx in self.linlora_layer_indices:  # Updated variable name
                # Calculate the modified weights
                w_combined = self.calc_new_w(idx)
                hidden_state = torch.matmul(hidden_state, w_combined.T)
                if layer.bias is not None:
                    hidden_state += layer.bias
            else:
                # Use the base layer directly
                hidden_state = layer(hidden_state)
            # Apply activation
            hidden_state = self.activation(hidden_state)
        # Pass through the output layer
        output = self.base_layers[-1](hidden_state)
        return output

    def calc_new_w(self, idx):
        if self.enforce_sv_positive:
            adapter = torch.relu(self.adapters[idx])
        else:
            adapter = self.adapters[idx]

        adapter = torch.diag(adapter)  # Diagonal matrix from adapter parameters
        scaling_adapter = torch.diag(self.scaling_adapters[idx])  # Diagonal scaling matrix
        # Compute the adapter
        adapter_matrix = self.U_directions[idx] @ adapter @ self.V_directions[idx] @ scaling_adapter
        # Combine with base weights
        base_weights = self.base_layers[idx].weight
        return base_weights + self.scaling_factor * adapter_matrix

    def get_matrix_U_and_V(self, weight_matrix, rank):
        # Compute SVD
        U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
        # Get indices of smallest singular values
        sorted_indices = torch.argsort(S)
        smallest_r_indices = sorted_indices[:rank]
        # Extract corresponding singular vectors
        U_r = U[:, smallest_r_indices]
        V_r = Vh[smallest_r_indices, :]
        return U_r, V_r

class UILinLoRAAdapter(nn.Module):
    def __init__(self, base_model, ranks=4, scaling_factor=1, linlora_layer_indices=[0], activation=nn.ReLU(), enforce_sv_positive=True):
        super(UILinLoRAAdapter, self).__init__()
        self.base_model = copy.deepcopy(base_model)
        self.scaling_factor = scaling_factor
        self.linlora_layer_indices = linlora_layer_indices  # Renamed for consistency
        self.activation = activation
        self.enforce_sv_positive = enforce_sv_positive

        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.eval()  # Keep the base model in evaluation mode

        # Prepare to store adapters for each layer
        self.adapters = nn.ParameterList()
        self.Dscaling_adapters = nn.ParameterList()
        self.Escaling_adapters = nn.ParameterList()
        self.U_directions = []
        self.V_directions = []

        # Collect layers from the base model
        self.base_layers = self.base_model.layers  # Assuming base_model.layers is nn.ModuleList

        # Ensure ranks is a list matching the number of layers
        if isinstance(ranks, int):
            ranks = [ranks] * len(self.base_layers)

        # Initialize adapters and singular vectors for specified layers
        for idx, (layer, rank) in enumerate(zip(self.base_layers[:-1], ranks)):  # Exclude the output layer
            if idx in self.linlora_layer_indices:  # Updated variable name
                weight = layer.weight.data
                # Compute SVD
                U_r, V_r = self.get_matrix_U_and_V(weight, rank)
                self.U_directions.append(U_r)
                self.V_directions.append(V_r)
                # Initialize adapter parameters
                adapter = nn.Parameter(torch.ones(rank) * 1e-7)
                Dscaling_adapter = nn.Parameter(torch.ones(weight.shape[1]))
                Escaling_adapter = nn.Parameter(torch.ones(weight.shape[0]))
                self.adapters.append(adapter)
                self.Dscaling_adapters.append(Dscaling_adapter)
                self.Escaling_adapters.append(Escaling_adapter)
            else:
                # Placeholders for layers without DLinLoRA
                self.U_directions.append(None)
                self.V_directions.append(None)
                self.adapters.append(None)
                self.scaling_adapters.append(None)

    def forward(self, x):
        hidden_state = x
        for idx, layer in enumerate(self.base_layers[:-1]):  # Exclude the output layer
            if idx in self.linlora_layer_indices:  # Updated variable name
                # Calculate the modified weights
                w_combined = self.calc_new_w(idx)
                hidden_state = torch.matmul(hidden_state, w_combined.T)
                if layer.bias is not None:
                    hidden_state += layer.bias
            else:
                # Use the base layer directly
                hidden_state = layer(hidden_state)
            # Apply activation
            hidden_state = self.activation(hidden_state)
        # Pass through the output layer
        output = self.base_layers[-1](hidden_state)
        return output

    def calc_new_w(self, idx):
        if self.enforce_sv_positive:
            adapter = torch.relu(self.adapters[idx])
        else:
            adapter = self.adapters[idx]
        adapter = torch.diag(self.adapters[idx])  # Diagonal matrix from adapter parameters
        Dscaling_adapter = torch.diag(self.Dscaling_adapters[idx])  # Diagonal scaling matrix
        Escaling_adapter = torch.diag(self.Escaling_adapters[idx])  # Diagonal scaling matrix
        # Compute the adapter
        adapter_matrix = Escaling_adapter @ self.U_directions[idx] @ adapter @ self.V_directions[idx] @ Dscaling_adapter
        # Combine with base weights
        base_weights = self.base_layers[idx].weight
        return base_weights + self.scaling_factor * adapter_matrix

    def get_matrix_U_and_V(self, weight_matrix, rank):
        # Compute SVD
        U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=False)
        # Get indices of smallest singular values
        sorted_indices = torch.argsort(S)
        smallest_r_indices = sorted_indices[:rank]
        # Extract corresponding singular vectors
        U_r = U[:, smallest_r_indices]
        V_r = Vh[smallest_r_indices, :]
        return U_r, V_r


class ArAdapter(nn.Module):
    def __init__(self, base_model, device, ranks=4, Ar_layer_indices=[0], activation=nn.ReLU(), scaling_factor=1):
        super(ArAdapter, self).__init__()
        self.base_model = copy.deepcopy(base_model)
        self.Ar_layer_indices = Ar_layer_indices 
        self.activation = activation
        self.device = device
        self.scaling_factor = scaling_factor

        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.eval()  # Keep the base model in evaluation mode

        # Prepare to store adapters for each layer
        self.Ar = nn.ModuleList()
        self.U_directions = []
        self.V_directions = []
        self.S_values = []

        # Collect layers from the base model
        self.base_layers = self.base_model.layers  # Assuming base_model.layers is nn.ModuleList

        # Ensure ranks is a list matching the number of layers
        if isinstance(ranks, int):
            ranks = [ranks] * len(self.base_layers)

        # Initialize adapters and singular vectors for specified layers
        for idx, (layer, rank) in enumerate(zip(self.base_layers[:-1], ranks)):  # Exclude the output layer
            if idx in self.Ar_layer_indices:  # Updated variable name
                weight = layer.weight.data
                # Compute SVD
                U, S, V = self.get_matrix_U_and_V(weight)
                self.U_directions.append(U)
                self.V_directions.append(V)
                self.S_values.append(S)
                # Initialize adapter parameters
                self.Ar.append(nn.Linear(V.shape[1], rank, bias=False))
                nn.init.normal_(self.Ar[-1].weight, mean=0.0, std=0.1)
            else:
                # Placeholders for layers without DLinLoRA
                self.U_directions.append(None)
                self.V_directions.append(None)
                self.S_values.append(None)

    def forward(self, x):
        hidden_state = x
        for idx, layer in enumerate(self.base_layers[:-1]):  # Exclude the output layer
            if idx in self.Ar_layer_indices:  # Updated variable name
                # Calculate the modified weights
                w_combined = self.calc_new_w(idx)
                hidden_state = torch.matmul(hidden_state, w_combined.T)
                if layer.bias is not None:
                    hidden_state += layer.bias
            else:
                # Use the base layer directly
                hidden_state = layer(hidden_state)
            # Apply activation
            hidden_state = self.activation(hidden_state)
        # Pass through the output layer
        output = self.base_layers[-1](hidden_state)
        return output

    def calc_new_w(self, idx):
        adapter_matrix = self.calc_new_adapter(idx)
        base_weights = self.base_layers[idx].weight
        return base_weights + self.scaling_factor*adapter_matrix
    
    def calc_new_adapter(self, idx):
        A_r = self.Ar[idx]  # Diagonal matrix from adapter parameters
        R = min(self.V_directions[idx].shape[0]-A_r.weight.shape[0], self.U_directions[idx].shape[0]-A_r.weight.shape[0])
        Sigma_R = torch.zeros(self.U_directions[idx].shape[0] - A_r.weight.shape[0], A_r.weight.shape[1], device=self.device)

        Sigma_R[:R, :R] = torch.diag(self.S_values[idx][:R])
        A_r = torch.cat([Sigma_R, A_r.weight], 0)
        return self.U_directions[idx] @ A_r @ self.V_directions[idx]
    

    def get_matrix_U_and_V(self, weight_matrix):
        U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=True)
        return U, S, Vh
    

class KernelAdapter(nn.Module):
    def __init__(self, base_model, device, ranks=4, scaling_factor=1, linlora_layer_indices=[0], activation=nn.ReLU(), enforce_sv_positive=True):
        super(KernelAdapter, self).__init__()
        self.base_model = copy.deepcopy(base_model)
        self.scaling_factor = scaling_factor
        self.linlora_layer_indices = linlora_layer_indices 
        self.activation = activation
        self.ranks = ranks
        self.device = device
        self.full_ranks = []
        self.enforce_sv_positive = enforce_sv_positive


        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.eval()  # Keep the base model in evaluation mode

        # Prepare to store adapters for each layer
        self.adapters = nn.ParameterList()
        self.left_unitary = nn.ParameterList()
        self.right_unitary = nn.ParameterList()
        self.U_directions = []
        self.V_directions = []

        # Collect layers from the base model
        self.base_layers = self.base_model.layers  # Assuming base_model.layers is nn.ModuleList

        # Ensure ranks is a list matching the number of layers
        if isinstance(ranks, int):
            self.ranks = [ranks] * len(self.base_layers)

        # Initialize adapters and singular vectors for specified layers
        for idx, (layer, rank) in enumerate(zip(self.base_layers[:-1], self.ranks)):  # Exclude the output layer
            if idx in self.linlora_layer_indices:  # Updated variable name
                weight = layer.weight.data
                self.full_ranks.append(min(weight.shape[0], weight.shape[1]))
                # Compute SVD
                U, V = self.get_matrix_U_and_V(weight)
                self.U_directions.append(U)
                self.V_directions.append(V)
                # Initialize adapter parameters
                adapter = nn.Parameter(torch.ones(rank) * 1e-7)

                # # Corrected way to apply orthogonal parameterization
                # left_unitary = torch.nn.utils.parametrizations.orthogonal(nn.Linear(self.U_directions[idx].shape[0], self.U_directions[idx].shape[0], bias=False))
                # right_unitary = torch.nn.utils.parametrizations.orthogonal(nn.Linear(self.V_directions[idx].shape[0], self.V_directions[idx].shape[0], bias=False))
                rank_to_preserve = self.full_ranks[idx] - rank
                left_orthogonal_size = self.U_directions[idx].shape[0] - rank_to_preserve
                right_orthogonal_size = self.V_directions[idx].shape[0] - rank_to_preserve

                left_orthogonal = torch.nn.utils.parametrizations.orthogonal(nn.Linear(left_orthogonal_size, left_orthogonal_size, bias=False))
                right_orthogonal = torch.nn.utils.parametrizations.orthogonal(nn.Linear(right_orthogonal_size, right_orthogonal_size, bias=False))
                
                self.adapters.append(adapter)
                self.left_unitary.append(left_orthogonal)
                self.right_unitary.append(right_orthogonal)
            else:
                # Placeholders for layers without DLinLoRA
                self.U_directions.append(None)
                self.V_directions.append(None)
                self.adapters.append(None)

    def forward(self, x):
        hidden_state = x
        for idx, layer in enumerate(self.base_layers[:-1]):  # Exclude the output layer
            if idx in self.linlora_layer_indices:  # Updated variable name
                # Calculate the modified weights
                w_combined = self.calc_new_w(idx)
                hidden_state = torch.matmul(hidden_state, w_combined.T)
                if layer.bias is not None:
                    hidden_state += layer.bias
            else:
                # Use the base layer directly
                hidden_state = layer(hidden_state)
            # Apply activation
            hidden_state = self.activation(hidden_state)
        # Pass through the output layer
        output = self.base_layers[-1](hidden_state)
        return output

    def calc_new_w(self, idx):
        left_size, right_size = self.base_layers[idx].weight.shape  
        left_unitary_for_calc = self.calc_left_unitary(idx, left_size)
        right_unitary_for_calc = self.calc_right_unitary(idx, right_size)
        sigma_matrix_for_calc = self.calc_sigma(idx, left_unitary_for_calc.shape[0], right_unitary_for_calc.shape[0])
        # adapter_matrix = self.U_directions[idx] @ left_unitary_for_calc @ sigma_matrix_for_calc @ right_unitary_for_calc.T @ self.V_directions[idx] # Guy Version
        adapter_matrix = left_unitary_for_calc @ self.U_directions[idx] @ sigma_matrix_for_calc @ self.V_directions[idx] @ right_unitary_for_calc.T # Shai Version
        base_weights = self.base_layers[idx].weight
        return base_weights + self.scaling_factor * adapter_matrix
    
    def calc_left_unitary(self, idx, left_size):
        rank_to_preserve = self.full_ranks[idx] - self.ranks[idx]
        proj_matrix = self.left_unitary[idx].weight
        return self._build_projection_matrix(proj_matrix, left_size, rank_to_preserve)
    
    def calc_right_unitary(self, idx, right_size):
        rank_to_preserve = self.full_ranks[idx] - self.ranks[idx]
        proj_matrix = self.right_unitary[idx].weight
        return self._build_projection_matrix(proj_matrix, right_size, rank_to_preserve)
    
    def _build_projection_matrix(self, projection_matrix, size, rank_to_preserve):
        upper_matrix = torch.eye(rank_to_preserve, rank_to_preserve, device=self.device)
        upper_matrix = torch.cat((upper_matrix, torch.zeros(rank_to_preserve, size - rank_to_preserve, device=self.device)), dim=1)
        down_matrix = torch.cat((torch.zeros(size - rank_to_preserve, rank_to_preserve, device=self.device), projection_matrix), dim=1)
        return torch.cat((upper_matrix, down_matrix), dim=0)

    
    def calc_sigma(self, idx, left_size, right_size):
        max_rank = min(left_size, right_size)
        not_trainable_part_size = max_rank - self.ranks[idx]
        sigma = torch.zeros(max_rank, max_rank, device=self.device)
        sigma[:not_trainable_part_size, :not_trainable_part_size] = torch.eye(not_trainable_part_size, not_trainable_part_size, device=self.device)
        # sigma[:not_trainable_part_size, :not_trainable_part_size] = torch.zeros(not_trainable_part_size, not_trainable_part_size, device=self.device)
        if self.enforce_sv_positive:
            non_trainable = torch.diag(torch.relu(self.adapters[idx]))
        else:
            non_trainable = torch.diag(self.adapters[idx])
        sigma[not_trainable_part_size:, not_trainable_part_size:] = non_trainable
        
        if max_rank < left_size:
            sigma = torch.cat((sigma, torch.zeros(left_size - max_rank, max_rank, device=self.device)), dim=0)
        elif max_rank < right_size:
            sigma = torch.cat((sigma, torch.zeros(max_rank, right_size - max_rank, device=self.device)), dim=1)
        
        return sigma


    def get_matrix_U_and_V(self, weight_matrix):
        U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=True)
        return U, Vh
    

def get_cifar10_datasets(device):
    # 1) Load CIFAR-10 train and test sets
    #    We'll use ToTensor() to convert PIL images to [C, H, W] tensors automatically.
    cifar10_train = datasets.CIFAR10(
        root="data", train=True, download=True, transform=None
    )
    cifar10_test = datasets.CIFAR10(
        root="data", train=False, download=True, transform=None
    )

    # 2) Convert train and test data to tensors 
    #    CIFAR-10 dataset.data is a NumPy array of shape (N, 32, 32, 3).
    #    We'll move the channel dimension to the front and convert to float.
    X_train = torch.tensor(cifar10_train.data).permute(0, 3, 1, 2).float() / 255.0
    y_train = torch.tensor(cifar10_train.targets)

    X_test = torch.tensor(cifar10_test.data).permute(0, 3, 1, 2).float() / 255.0
    y_test = torch.tensor(cifar10_test.targets)

    # Move them initially to CPU (we will .to(device) later once we split).
    # This allows easy slicing before pushing to the GPU (especially for train_test_split).
    # If you prefer, you can move them to device at the end all at once.

    # 3) Split pretrain (labels 0-8) and finetune (label 9) data
    #    Exactly as you did for MNIST.
    X_pretrain_train = X_train[y_train < 9]
    y_pretrain_train = y_train[y_train < 9]

    X_finetune_train = X_train[y_train == 9]
    y_finetune_train = y_train[y_train == 9]

    # Combine pretrain and finetune sets for a “train on all classes” scenario
    X_pre_ft_train = torch.cat([X_pretrain_train, X_finetune_train], dim=0)
    y_pre_ft_train = torch.cat([y_pretrain_train, y_finetune_train], dim=0)

    # 4) Split the test set (80%/20%), then separate those 20% into pretrain/finetune subsets
    X_test_combined_pre_fine, X_test_combined, y_test_combined_pre_fine, y_test_combined = train_test_split(
        X_test, y_test, test_size=0.2, random_state=42, stratify=y_test
    )

    X_pretrain_test = X_test_combined_pre_fine[y_test_combined_pre_fine < 9]
    y_pretrain_test = y_test_combined_pre_fine[y_test_combined_pre_fine < 9]

    X_finetune_test = X_test_combined_pre_fine[y_test_combined_pre_fine == 9]
    y_finetune_test = y_test_combined_pre_fine[y_test_combined_pre_fine == 9]

    X_pre_ft_test = torch.cat([X_pretrain_test, X_finetune_test], dim=0)
    y_pre_ft_test = torch.cat([y_pretrain_test, y_finetune_test], dim=0)

    # 5) Combined test set (the 80% portion)
    X_combined_test = X_test_combined
    y_combined_test = y_test_combined

    # 6) Flatten the data for linear layers: 
    #    CIFAR-10 images are 3×32×32, so each sample becomes a 3072-dim vector.
    X_pretrain_train = X_pretrain_train.view(X_pretrain_train.size(0), -1)
    X_finetune_train = X_finetune_train.view(X_finetune_train.size(0), -1)
    X_pre_ft_train   = X_pre_ft_train.view(X_pre_ft_train.size(0), -1)

    X_pretrain_test  = X_pretrain_test.view(X_pretrain_test.size(0), -1)
    X_finetune_test  = X_finetune_test.view(X_finetune_test.size(0), -1)
    X_pre_ft_test    = X_pre_ft_test.view(X_pre_ft_test.size(0), -1)

    X_combined_test  = X_combined_test.view(X_combined_test.size(0), -1)

    # 7) Move all tensors to the specified device (GPU/CPU)
    X_pretrain_train = X_pretrain_train.to(device)
    y_pretrain_train = y_pretrain_train.to(device)

    X_finetune_train = X_finetune_train.to(device)
    y_finetune_train = y_finetune_train.to(device)

    X_pre_ft_train   = X_pre_ft_train.to(device)
    y_pre_ft_train   = y_pre_ft_train.to(device)

    X_pretrain_test  = X_pretrain_test.to(device)
    y_pretrain_test  = y_pretrain_test.to(device)

    X_finetune_test  = X_finetune_test.to(device)
    y_finetune_test  = y_finetune_test.to(device)

    X_pre_ft_test    = X_pre_ft_test.to(device)
    y_pre_ft_test    = y_pre_ft_test.to(device)

    X_combined_test  = X_combined_test.to(device)
    y_combined_test  = y_combined_test.to(device)

    # 8) Create DataLoaders
    batch_size = 512
    pretrain_train_loader = DataLoader(
        TensorDataset(X_pretrain_train, y_pretrain_train),
        batch_size=batch_size, shuffle=True
    )
    finetune_train_loader = DataLoader(
        TensorDataset(X_finetune_train, y_finetune_train),
        batch_size=batch_size, shuffle=True
    )
    pre_ft_train_loader = DataLoader(
        TensorDataset(X_pre_ft_train, y_pre_ft_train),
        batch_size=batch_size, shuffle=True
    )

    pretrain_test_loader = DataLoader(
        TensorDataset(X_pretrain_test, y_pretrain_test),
        batch_size=batch_size, shuffle=False
    )
    finetune_test_loader = DataLoader(
        TensorDataset(X_finetune_test, y_finetune_test),
        batch_size=batch_size, shuffle=False
    )
    pre_ft_test_loader = DataLoader(
        TensorDataset(X_pre_ft_test, y_pre_ft_test),
        batch_size=batch_size, shuffle=False
    )
    combined_test_loader = DataLoader(
        TensorDataset(X_combined_test, y_combined_test),
        batch_size=batch_size, shuffle=False
    )

    # 9) Return the DataLoaders
    return (
        pretrain_train_loader,
        finetune_train_loader,
        pretrain_test_loader,
        finetune_test_loader,
        combined_test_loader,
        pre_ft_train_loader,
        pre_ft_test_loader,
    )


class UIOrthoLoRAV2(nn.Module):
    def __init__(self, base_model, device, sigma_regularization, num_of_svectors_to_adapt,
                 num_of_svalues_to_adapt=4, scaling_factor=1, 
                 E_init_value=1e-7, D_init_value=1e-7, adapter_init_value=1e-7,
                 layers_to_adapt=[0], activation=nn.ReLU(), enforce_sv_positive=False):
        super(UIOrthoLoRAV2, self).__init__()
        self.base_model = copy.deepcopy(base_model)
        self.scaling_factor = scaling_factor
        self.layers_to_adapt = layers_to_adapt
        self.activation = activation
        self.device = device
        self.full_ranks = []
        self.enforce_sv_positive = enforce_sv_positive
        self.sigma_regularization = sigma_regularization  # now a PyTorch vector
        self.num_of_svectors_to_adapt = num_of_svectors_to_adapt
        self.num_of_svalues_to_adapt = num_of_svalues_to_adapt
        self.adapter_init_value = adapter_init_value
        self.E_init_value = E_init_value
        self.D_init_value = D_init_value

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        # Prepare containers for adapters and extra parameters
        self.adapters = nn.ParameterList()
        self.left_unitary = nn.ParameterList()
        self.right_unitary = nn.ParameterList()
        self.diag_left = nn.ParameterList()
        self.diag_right = nn.ParameterList()
        self.U_directions = []
        self.V_directions = []

        self.base_layers = self.base_model.layers

        # Initialize adapters only for layers in layers_to_adapt (only one layer in our case)
        for idx, (layer, rank) in enumerate(zip(self.base_layers[:-1], self.num_of_svalues_to_adapt)):
            if idx in self.layers_to_adapt:
                weight = layer.weight.data
                self.full_ranks.append(min(weight.shape[0], weight.shape[1]))
                U, V = self.get_matrix_U_and_V(weight)
                self.U_directions.append(U)
                self.V_directions.append(V)
                adapter = nn.Parameter(torch.ones(rank) * self.adapter_init_value)
                
                rank_to_preserve = self.full_ranks[idx] - self.num_of_svectors_to_adapt[idx]
                orthogonal_size = min(U.shape[0], V.shape[0]) - rank_to_preserve
                # left_orthogonal_size = U.shape[0] - rank_to_preserve
                # right_orthogonal_size = V.shape[0] - rank_to_preserve

                left_orthogonal = torch.nn.utils.parametrizations.orthogonal(
                    nn.Linear(orthogonal_size, orthogonal_size, bias=False)
                )
                right_orthogonal = torch.nn.utils.parametrizations.orthogonal(
                    nn.Linear(orthogonal_size, orthogonal_size, bias=False)
                )
                
                self.adapters.append(adapter)
                self.left_unitary.append(left_orthogonal)
                self.right_unitary.append(right_orthogonal)
                
                left_size = weight.shape[0]
                right_size = weight.shape[1]
                E_diag_left = nn.Parameter(torch.ones(left_size) * self.E_init_value)
                D_diag_right = nn.Parameter(torch.ones(right_size) * self.D_init_value)
                self.diag_left.append(E_diag_left)
                self.diag_right.append(D_diag_right)
            else:
                self.U_directions.append(None)
                self.V_directions.append(None)
                self.adapters.append(None)
                self.diag_left.append(None)
                self.diag_right.append(None)

    def forward(self, x):
        hidden_state = x
        for idx, layer in enumerate(self.base_layers[:-1]):
            if idx in self.layers_to_adapt:
                w_combined = self.calc_new_w(idx)
                hidden_state = torch.matmul(hidden_state, w_combined.T)
                if layer.bias is not None:
                    hidden_state += layer.bias
            else:
                hidden_state = layer(hidden_state)
            hidden_state = self.activation(hidden_state)
        output = self.base_layers[-1](hidden_state)
        return output

    def calc_new_w(self, idx):
        left_size, right_size = self.base_layers[idx].weight.shape
        left_unitary_for_calc = self.calc_left_unitary(idx, left_size)
        right_unitary_for_calc = self.calc_right_unitary(idx, right_size)
        sigma_matrix_for_calc = self.calc_sigma(idx, left_unitary_for_calc.shape[0], right_unitary_for_calc.shape[0])
        adapter_matrix = left_unitary_for_calc @ self.U_directions[idx] @ sigma_matrix_for_calc @ self.V_directions[idx] @ right_unitary_for_calc.T
        
        if self.diag_left[idx] is not None and self.diag_right[idx] is not None:
            left_diag_matrix = torch.diag(self.diag_left[idx])
            right_diag_matrix = torch.diag(self.diag_right[idx])
            adapter_matrix = left_diag_matrix @ adapter_matrix @ right_diag_matrix
            
        base_weights = self.base_layers[idx].weight
        return base_weights + self.scaling_factor * adapter_matrix

    def calc_left_unitary(self, idx, left_size):
        # rank_to_preserve = self.full_ranks[idx] - self.num_of_svectors_to_adapt[idx]
        rank_to_preserve = left_size - self.num_of_svectors_to_adapt[idx]
        proj_matrix = self.left_unitary[idx].weight
        return self._build_projection_matrix(proj_matrix, left_size, rank_to_preserve)
    
    def calc_right_unitary(self, idx, right_size):
        # rank_to_preserve = self.full_ranks[idx] - self.num_of_svectors_to_adapt[idx]
        rank_to_preserve = right_size - self.num_of_svectors_to_adapt[idx]
        proj_matrix = self.right_unitary[idx].weight
        return self._build_projection_matrix(proj_matrix, right_size, rank_to_preserve)
    
    def _build_projection_matrix(self, projection_matrix, size, rank_to_preserve):
        upper_matrix = torch.eye(rank_to_preserve, rank_to_preserve, device=self.device)
        upper_matrix = torch.cat((upper_matrix, torch.zeros(rank_to_preserve, size - rank_to_preserve, device=self.device)), dim=1)
        down_matrix = torch.cat((torch.zeros(size - rank_to_preserve, rank_to_preserve, device=self.device), projection_matrix), dim=1)
        return torch.cat((upper_matrix, down_matrix), dim=0)
    
    def calc_sigma(self, idx, left_size, right_size):
        max_rank = min(left_size, right_size)
        not_trainable_part_size = max_rank - self.num_of_svalues_to_adapt[idx]
        sigma = torch.zeros(max_rank, max_rank, device=self.device)
        sigma[:not_trainable_part_size, :not_trainable_part_size] = torch.eye(not_trainable_part_size, device=self.device)
        if self.enforce_sv_positive:
            non_trainable = torch.diag(torch.relu(self.adapters[idx]))
        else:
            non_trainable = torch.diag(self.adapters[idx])
        sigma[not_trainable_part_size:, not_trainable_part_size:] = non_trainable
        
        if max_rank < left_size:
            sigma = torch.cat((sigma, torch.zeros(left_size - max_rank, max_rank, device=self.device)), dim=0)
        elif max_rank < right_size:
            sigma = torch.cat((sigma, torch.zeros(max_rank, right_size - max_rank, device=self.device)), dim=1)
        
        # Use sigma_regularization as a vector to build a diagonal matrix
        reg_matrix = torch.diag(self.sigma_regularization)
        return reg_matrix @ sigma

    def get_matrix_U_and_V(self, weight_matrix):
        U, S, Vh = torch.linalg.svd(weight_matrix, full_matrices=True)
        return U, Vh
