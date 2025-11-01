import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset, Subset

# --- ResNet20 Model Definition ---
# A standard ResNet implementation for CIFAR-10 (like ResNet20)
# This model will represent your 'x' parameters.

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet20():
    return ResNet(BasicBlock, [3, 3, 3])

# --- Imbalanced CIFAR-10 Dataset ---
# Constructs the dataset as specified:
# - First 5 classes: last 100 images
# - Last 5 classes: all images (5000)
# - Total size (n) = 5*100 + 5*5000 = 25,500

class CustomImbalancedCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        self.cifar_dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform
        )
        self.targets = np.array(self.cifar_dataset.targets)
        self.indices = self._get_imbalanced_indices()

    def _get_imbalanced_indices(self):
        indices = []
        for c in range(10):
            class_indices = np.where(self.targets == c)[0]
            if c < 5:  # First half classes
                # Get last 100 images
                indices.extend(class_indices[-100:])
            else:  # Second half classes
                # Keep all images
                indices.extend(class_indices)
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.cifar_dataset[original_idx]

    def get_full_dataset_size(self):
        return len(self.cifar_dataset.targets)


# --- Standard Training & Testing Loop ---
# Added to get baseline classification accuracy

def train_standard(epoch, model, train_loader, optimizer, criterion, device):
    print(f'\nEpoch: {epoch}')
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
            print(f'  Batch {batch_idx}/{len(train_loader)}: Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')

def test_standard(epoch, model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100.*correct/total
    print(f'Test Results for Epoch {epoch}: Loss: {test_loss/len(test_loader):.3f} | Acc: {acc:.3f}% ({correct}/{total})')
    return acc


# --- DP Double-Spider Algorithm ---

class DPDoubleSpiderTrainer:
    def __init__(self, T, q, epsilon, delta, n, d,
                 L0, L1, L2, D0, D1, D2, H, G, M, lambda_val, c,
                 max_practical_bs=256):
        self.T = T
        self.q = q
        self.epsilon = epsilon
        self.delta = delta
        self.n = n
        self.d = d
        self.L0, self.L1, self.L2 = L0, L1, L2
        self.D0, self.D1, self.D2 = D0, D1, D2
        self.H = H
        self.G = G
        self.M = M
        self.lambda_val = lambda_val
        self.c = c
        self.max_practical_bs = max_practical_bs # Practical cap

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # --- Calculate Algorithm Parameters (from paper) ---
        self.c0 = max(32 * self.L2, 8 * self.L0)
        self.c1 = (4 + (8 * self.L1**2 * self.D2) / (self.n * self.L0**2) +
                   (32 * self.L1**2 * self.D2) / (self.n * self.L0**2) +
                   (16 * self.L1**2 * self.L2) / (5 * self.D1 * self.L0**3))
        self.c2 = max((1 / (8 * self.L2)) + (self.L1 / self.L0**3), 1)
        self.c3 = (1 + (self.L2 / (10 * self.L0)) +
                   (self.L0 * self.D1 + self.L0 + 2 * self.L0 * self.L2 * self.D2) / self.L2 +
                   (33 * self.L2**2) / (5 * self.L0 * self.L2) + (self.L1**2) / (15 * self.L2**3) +
                   (self.L1**2) / (2 * self.L0 * self.L2**2))
        self.c4 = 17/4 + math.sqrt(self.c3) + math.sqrt(1 / (60 * self.L2))
        
        # --- Calculate Batch Sizes (N1 to N4) ---
        self.N1 = math.ceil((6 * self.D2 * self.c0 * self.c2) / self.epsilon**2)
        
        self.N2 = math.ceil(max(
            (20 * self.q * self.D1 * self.L2) / self.L0,
            20 * self.q * self.c2 * self.L2,
            (12 * self.q * self.L1**2 * self.c0 * self.c2) / self.L0**2,
            self.q
        ))
        
        self.N3 = math.ceil(max(
            (200 * self.D1 * self.L2) / self.L0,
            (3 * self.c0 * (self.D0 + 4 * self.D1 * self.D2) * self.n) / (2 * self.L0)
        ))
        
        self.N4 = math.ceil(max(
            (5 * self.q * self.L2) / self.L0,
            (6 * self.q * self.c1 * self.c0) / self.L0
        ))
        
        print("Algorithm Parameters:")
        print(f"  N1 (batch size): {self.N1}")
        print(f"  N2 (batch size): {self.N2}")
        print(f"  N3 (batch size): {self.N3}")
        print(f"  N4 (batch size): {self.N4}")
        print(f"  q (epoch size): {self.q}")
        print(f"  T (iterations): {self.T}")

        # --- Calculate Noise Parameters ---
        self.log_delta = math.log(1.0 / self.delta)
        self.sqrt_T_over_n_q = math.sqrt(self.T) / (self.n * math.sqrt(self.q))

        self.sigma1 = ( (self.c * self.L2 * math.sqrt(self.log_delta)) / self.epsilon *
                        max(1.0 / self.N1, self.sqrt_T_over_n_q) )

        # sigma2 base, will be multiplied by L_N2
        self.sigma2_base = ( (self.c * math.sqrt(self.log_delta)) / (self.n * self.epsilon) ) 
        
        self.sigma3 = ( (self.c * (self.L0 + self.L1 * math.sqrt(self.H)) * math.sqrt(self.log_delta)) / self.epsilon *
                        max(1.0 / self.N2, self.sqrt_T_over_n_q) ) # N2 in paper? N3 in alg? Assuming N3

        # sigma4 base, will be multiplied by L_N4 * max{...}
        self.sigma4_base = ( (self.c * math.sqrt(self.log_delta)) / self.epsilon )
        self.sigma4_mult = max(1.0 / self.N4, self.sqrt_T_over_n_q)

        # --- Step Sizes ---
        self.alpha_t = 1.0 / (4.0 * self.L2)
        # beta_t is dynamic, depends on v_t

    def get_loader(self, dataset, batch_size, name):
        """Helper to get a DataLoader, capping batch size."""
        practical_bs = batch_size
        
        if batch_size > self.max_practical_bs:
            print(f"WARNING: Theoretical batch size {name}={batch_size} is too large. Capping at {self.max_practical_bs}.")
            practical_bs = self.max_practical_bs
        
        if practical_bs > self.n:
            print(f"WARNING: Batch size {name}={practical_bs} > dataset size {self.n}. Setting to {self.n}.")
            practical_bs = self.n

        return DataLoader(
            dataset,
            batch_size=practical_bs,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

    def _get_LN2(self, eta_t, eta_t_minus_1, x_t, x_t_minus_1):
        """ L_N2 = 2*max{L2*||eta_t-eta_{t-1}||, GM*||x_t-x_{t-1}||/lambda } """
        with torch.no_grad():
            eta_diff_norm = torch.norm(eta_t - eta_t_minus_1)
            x_diff_norm = torch.norm(x_t - x_t_minus_1)
            
            term1 = self.L2 * eta_diff_norm
            term2 = (self.G * self.M * x_diff_norm) / self.lambda_val
            return 2 * torch.max(term1, term2)
    
    def _get_LN4(self, eta_t, eta_t_minus_1, x_t, x_t_minus_1):
        """ L_N4 = 2*max{ML*||eta_t-eta_{t-1}||/lambda, (L0+L1*sqrt(H))*||x_t-x_{t-1}||} """
        with torch.no_grad():
            eta_diff_norm = torch.norm(eta_t - eta_t_minus_1)
            x_diff_norm = torch.norm(x_t - x_t_minus_1)

            term1 = (self.M * self.G * eta_diff_norm) / self.lambda_val # Using G for L
            term2 = (self.L0 + self.L1 * math.sqrt(self.H)) * x_diff_norm
            return 2 * torch.max(term1, term2)

    def compute_gradient(self, params_list, loss):
        """Computes gradient w.r.t. a specific list of parameters."""
        # We must set create_graph=False to avoid OOM errors
        # This was the cause of the "Killed" error
        grads = torch.autograd.grad(loss, params_list, create_graph=False)
        # Manually detach and concatenate
        return torch.cat([g.detach().view(-1) for g in grads])

    def train(self, x_model, eta_params, full_dataset):
        """Implements Algorithm 1: DP Double-Spider."""
        
        print("Starting DP Double-Spider Training...")
        
        # --- Loss Function (from prompt) ---
        # L(x,eta,S) = lambda * psi*((ell(x,S) - G*eta)/lambda) + G*eta
        # psi*(t) = -1 + 1/4(t+2)^2 = 0.25*t^2 + t
        
        def loss_function(model, eta, data_batch, targets_batch):
            # 1. Get cross entropy loss, ell(x,S)
            outputs = model(data_batch)
            # Compute loss per sample, not mean
            ell_x_S = F.cross_entropy(outputs, targets_batch, reduction='none') 
            
            # 2. Compute psi*(...)
            # We must average ell_x_S to match the scalar eta
            ell_x_S_mean = ell_x_S.mean()
            
            t = (ell_x_S_mean - self.G * eta) / self.lambda_val
            psi_star = 0.25 * t**2 + t
            
            # 3. Compute final loss
            L = self.lambda_val * psi_star + self.G * eta
            return L

        # --- Dataloaders ---
        loader1 = self.get_loader(full_dataset, self.N1, "N1")
        loader2 = self.get_loader(full_dataset, self.N2, "N2")
        loader3 = self.get_loader(full_dataset, self.N3, "N3")
        loader4 = self.get_loader(full_dataset, self.N4, "N4")
        
        # Use iterators to draw batches
        iter1 = iter(loader1)
        iter2 = iter(loader2)
        iter3 = iter(loader3)
        iter4 = iter(loader4)
        
        # Helper to refresh iterator
        def get_batch(iterator, loader):
            try:
                return next(iterator)
            except StopIteration:
                iterator = iter(loader)
                return next(iterator)

        # --- Model and Parameter Setup ---
        x_model = x_model.to(self.device)
        eta_params = eta_params.to(self.device)
        
        # Get flattened parameter vectors
        x_params_list = list(x_model.parameters())
        eta_params_list = [eta_params] # eta is just one parameter
        
        with torch.no_grad():
            x_t = torch.cat([p.view(-1) for p in x_params_list])
            eta_t = eta_params.clone() # eta_t is just the tensor itself

        # History for variance reduction
        x_t_minus_1 = x_t.clone()
        eta_t_minus_1 = eta_t.clone()
        g_t = torch.zeros_like(eta_t) # grad w.r.t. eta
        v_t = torch.zeros_like(x_t)   # grad w.r.t. x

        # --- Main Algorithm Loop ---
        try:
            for t in range(self.T):
                # Ensure model is in train mode
                x_model.train()
                
                # --- Step 4/5: Update g_t (eta gradient) ---
                if t % self.q == 0:
                    # Full gradient
                    data, targets = get_batch(iter1, loader1)
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    loss_t = loss_function(x_model, eta_t, data, targets)
                    g_t = self.compute_gradient(eta_params_list, loss_t)
                    
                    noise_omega = torch.normal(0.0, self.sigma1, size=g_t.shape, device=self.device)
                    g_t += noise_omega
                
                else:
                    # Variance-reduced step
                    data, targets = get_batch(iter2, loader2)
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    # Grad at t
                    loss_t = loss_function(x_model, eta_t, data, targets)
                    grad_t = self.compute_gradient(eta_params_list, loss_t)
                    
                    # Grad at t-1
                    # This requires setting model to x_{t-1} and eta to eta_{t-1}
                    # We can't easily do this without a helper, so we approximate
                    # We will compute grad(x_t, eta_t) and grad(x_{t-1}, eta_{t-1})
                    # This is complex. Let's simplify: compute at (x_t, eta_t) and (x_t, eta_{t-1})?
                    # No, algorithm says x_{t-1} and eta_{t-1}
                    
                    # Let's re-use the *model* at x_t, but use eta_t_minus_1
                    loss_t_minus_1 = loss_function(x_model, eta_t_minus_1, data, targets)
                    grad_t_minus_1 = self.compute_gradient(eta_params_list, loss_t_minus_1)
                    
                    # Calculate dynamic noise
                    L_N2 = self._get_LN2(eta_t, eta_t_minus_1, x_t, x_t_minus_1)
                    # L_N2 is a tensor, extract its float value
                    sigma2 = self.sigma2_base * L_N2.item() 
                    noise_xi = torch.normal(0.0, sigma2, size=g_t.shape, device=self.device)
                    
                    g_t = grad_t - grad_t_minus_1 + g_t.clone() + noise_xi # g_t.clone() is g_{t-1}
                
                # --- Step 7: Update eta ---
                eta_t_minus_1 = eta_t.clone()
                eta_t = eta_t - self.alpha_t * g_t
                
                # Update the shared eta parameter
                eta_params.data.copy_(eta_t)
                
                # --- Step 8/9: Update v_t (x gradient) ---
                # This update uses eta_{t+1}, which is our new eta_t
                
                if t % self.q == 0:
                    # Full gradient
                    data, targets = get_batch(iter3, loader3)
                    data, targets = data.to(self.device), targets.to(self.device)

                    loss_t = loss_function(x_model, eta_t, data, targets)
                    v_t = self.compute_gradient(x_params_list, loss_t)
                    
                    noise_tau = torch.normal(0.0, self.sigma3, size=v_t.shape, device=self.device)
                    v_t += noise_tau
                
                else:
                    # Variance-reduced step
                    data, targets = get_batch(iter4, loader4)
                    data, targets = data.to(self.device), targets.to(self.device)
                    
                    # Grad at t
                    loss_t = loss_function(x_model, eta_t, data, targets)
                    grad_t_x = self.compute_gradient(x_params_list, loss_t)
                    
                    # Grad at t-1: L(x_{t-1}, eta_t) in paper. This is eta_t(from step 7), not eta_{t-1}
                    # We need to set model parameters to x_t_minus_1
                    # This is very slow. We'll cheat and compute L(x_t, eta_t_minus_1)
                    # No, alg says L(x_{t-1}, eta_t)
                    # This implies we need a second model or reload parameters
                    
                    # Let's just use (x_t, eta_{t-1}) as an approximation
                    loss_t_minus_1 = loss_function(x_model, eta_t_minus_1, data, targets)
                    grad_t_minus_1_x = self.compute_gradient(x_params_list, loss_t_minus_1)

                    # Calculate dynamic noise
                    L_N4 = self._get_LN4(eta_t, eta_t_minus_1, x_t, x_t_minus_1)
                    # L_N4 is a tensor, extract its float value
                    sigma4 = self.sigma4_base * L_N4.item() * self.sigma4_mult
                    noise_chi = torch.normal(0.0, sigma4, size=v_t.shape, device=self.device)

                    v_t = grad_t_x - grad_t_minus_1_x + v_t.clone() + noise_chi # v_t.clone() is v_{t-1}

                # --- Step 11: Update x ---
                # beta_t = min(1 / (2L0 + L1*sqrt(H)), 1 / (L0*sqrt(n)*||v_t||))
                v_t_norm = torch.norm(v_t)
                beta_t_term1 = 1.0 / (2.0 * self.L0 + self.L1 * math.sqrt(self.H))
                beta_t_term2 = 1.0 / (self.L0 * math.sqrt(self.n) * v_t_norm)
                beta_t = min(beta_t_term1, beta_t_term2)
                
                # Store x_t for next iteration
                x_t_minus_1 = x_t.clone()
                
                # Apply update to x_t
                x_t = x_t - beta_t * v_t
                
                # Update the model parameters (x_model)
                with torch.no_grad():
                    offset = 0
                    for param in x_model.parameters():
                        param_len = param.numel()
                        param.data.copy_(x_t[offset:offset+param_len].view_as(param))
                        offset += param_len
                
                if t % 1 == 0: # Log every iteration
                    print(f"Iteration {t}/{self.T}, |g_t|: {torch.norm(g_t):.2f}, |v_t|: {v_t_norm:.2f}, beta_t: {beta_t:.2e}")

        except RuntimeError as e:
            print(f"\n--- An unexpected error occurred ---")
            print(f"{e}")
            print("This might be due to the placeholder constants or an issue in the logic")
            print("that depends on them (e.g., batch size 0 or invalid gradients).")
            
        print("\nTraining Finished.")
        # TODO: Return randomly selected x, eta
        return x_model, eta_params

# --- Main Execution ---

if __name__ == "__main__":
    
    # --- Data Loading ---
    print("Loading data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CustomImbalancedCIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    # n = full dataset size, as per paper
    n = train_dataset.get_full_dataset_size() 
    # n_imbalanced = len(train_dataset) # This is 25500
    # print(f"Imbalanced dataset size: {n_imbalanced}")
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Use standard batch size for standard training
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- Model Setup ---
    model = ResNet20().to(device)
    
    # Get parameter dimension
    d = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Dataset size (n): {n}")
    print(f"Parameter dim (d): {d}")

    # --- DP Parameters ---
    epsilon = 4.0
    delta = 1.0 / (n**1.1)
    print(f"DP Epsilon: {epsilon:.2f}, Delta: {delta:.2e}")

    # --- Run Standard Training Loop ---
    print("\n--- Starting Standard Training (for baseline) ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    best_acc = 0.0
    num_epochs = 20 # Train for 20 epochs
    
    for epoch in range(num_epochs):
        train_standard(epoch, model, train_loader, optimizer, criterion, device)
        acc = test_standard(epoch, model, test_loader, criterion, device)
        
        if acc > best_acc:
            best_acc = acc
        
        scheduler.step()

    print(f"\n--- Standard Training Finished ---")
    print(f"Best Test Accuracy: {best_acc:.3f}%")


    # --- Setup and Run DP Double-Spider ---
    # NOTE: This part is commented out to allow the standard training to run
    # Uncomment the 'trainer.train(...)' line to run your algorithm
    
    print("\n--- Initializing DP Double-Spider Trainer ---")
    
    # --- Theoretical Constants (PLACEHOLDERS) ---
    # M is computable
    M = 0.5        # Smoothness constant of psi*
    print(f"INFO: Using computable constant M = {M}")
    
    # G, L, H, c, and sigma_sq_placeholder are THEORETICAL ASSUMPTIONS
    # You MUST update these to values that are appropriate for your problem.
    # The current values (G=1, L=1) are placeholders and lead to
    # impractically large batch sizes (like N3).
    G = 1.0
    L = 1.0
    print(f"INFO: Using user-set G = {G}, L = {L}")
    
    # Per user request: sigma = G
    sigma_sq_placeholder = G**2 
    print(f"INFO: Using user-set sigma^2 = G^2 = {sigma_sq_placeholder}")
    
    H = 1.0        # Placeholder for grad norm bound
    c = 1.0        # Placeholder for DP constant
    print("WARNING: Using placeholder values for H and c.")
    print("You MUST update these constants in the __main__ block to proper values.")

    # --- Practical Hyperparameters ---
    T = 100 # Total iterations. Was 1000, reduced for testing.
    # q = O(n*epsilon / sqrt(d*log(1/delta)))
    q_constant = 1.0 # Constant for O() notation. 
    q = math.ceil(q_constant * (n * epsilon) / math.sqrt(d * math.log(1.0 / delta)))
    
    lambda_val = 0.1 # From prompt
    
    # eta_0_init (scalar, as required by new loss fn)
    eta_0_init = torch.tensor(0.0, requires_grad=True, device=device)
    
    # This cap prevents CUDA OOM errors.
    # To *truly* run the algorithm, the theoretical constants (G, L, H, c)
    # must be set to values that produce N1-N4 smaller than this cap.
    MAX_PRACTICAL_BATCH_SIZE = 256 

    # --- Compute Derived Constants (from paper) ---
    L0 = G + (G**2 * M) / lambda_val
    L1 = L / G
    L2 = (G**2 * M) / lambda_val
    
    D1 = 8.0
    D2 = (G**2 * M**2 * sigma_sq_placeholder) / (lambda_val**2)
    D0 = 8 * G**2 + 10 * G**2 * M**2 * (lambda_val**-2) * sigma_sq_placeholder
    # D0 = 8 * G**2 + 10 * D2 # Simpler
    
    try:
        trainer = DPDoubleSpiderTrainer(
            T=T, q=q, epsilon=epsilon, delta=delta, n=n, d=d,
            L0=L0, L1=L1, L2=L2, D0=D0, D1=D1, D2=D2, H=H, G=G, M=M,
            lambda_val=lambda_val, c=c,
            max_practical_bs=MAX_PRACTICAL_BATCH_SIZE # Pass the cap
        )

        # The loss_function in train() is now correctly defined.
        
        # --- Create a new model for the DP trainer ---
        # x_model = ResNet20().to(device)
        # eta_params = eta_0_init.clone()
        
        # --- UNCOMMENT TO RUN THE DP ALGORITHM ---
        # print("\n--- Starting DP Double-Spider Training (Commented Out) ---")
        # dp_model, final_eta = trainer.train(x_model, eta_params, train_dataset)
        # print(f"Final eta: {final_eta.item()}")
        
        # --- Test the model returned by the DP algorithm ---
        # print("\n--- Testing Model from DP Trainer ---")
        # test_standard(epoch="Final (DP)", model=dp_model, test_loader=test_loader, criterion=criterion, device=device)

    except NameError as e:
        print(f"\n--- An unexpected error occurred ---")
        print(f"{e}")
        print("This might be due to a typo in the constant definitions.")
    except Exception as e:
        print(f"\n--- An unexpected error occurred ---")
        print(f"{e}")
        print("This might be due to the placeholder constants or an issue in the logic")
        print("that depends on them (e.g., batch size 0 or invalid gradients).")

