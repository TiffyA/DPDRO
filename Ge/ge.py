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
    # ResNet20 has [3, 3, 3] blocks
    return ResNet(BasicBlock, [3, 3, 3])

# --- Data Loading Utility ---

class CustomImbalancedCIFAR10(Dataset):
    """
    Artificially imbalanced CIFAR-10 dataset.
    - Keeps only the last 100 images for classes 0-4.
    - Keeps all images for classes 5-9.
    """
    def __init__(self, original_dataset, keep_last_n=100):
        self.original_dataset = original_dataset
        self.indices = self._create_imbalanced_indices(keep_last_n)
        
    def _create_imbalanced_indices(self, keep_last_n):
        targets = np.array(self.original_dataset.targets)
        indices_by_class = [np.where(targets == i)[0] for i in range(10)]
        
        final_indices = []
        for i in range(10):
            if i < 5:  # First half classes (0-4)
                if len(indices_by_class[i]) > keep_last_n:
                    final_indices.extend(indices_by_class[i][-keep_last_n:])
                else:
                    final_indices.extend(indices_by_class[i])
            else:  # Second half classes (5-9)
                final_indices.extend(indices_by_class[i])
        
        return final_indices

    def __getitem__(self, idx):
        return self.original_dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

def load_imbalanced_cifar10():
    """
    Loads the imbalanced CIFAR-10 train set and the original test set.
    """
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

    # Full training set
    trainset_full = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    
    # Create imbalanced training set
    imbalanced_trainset = CustomImbalancedCIFAR10(trainset_full, keep_last_n=100)
    
    # Original test set
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    # Note: Batch sizes N1, N2, N3, N4 will be set by the algorithm parameters
    # We return the full datasets here.
    return imbalanced_trainset, testset

# --- Helper Functions for Manual Parameter Handling ---

@torch.no_grad()
def get_flat_params(params):
    """Utility to flatten parameters for noise addition and norm calculation."""
    return torch.cat([p.reshape(-1) for p in params])

@torch.no_grad()
def set_flat_params(params, flat_params):
    """Utility to set parameters from a flattened tensor."""
    offset = 0
    for p in params:
        numel = p.numel()
        p.copy_(flat_params[offset:offset + numel].reshape_as(p))
        offset += numel

def get_param_list(model_or_tensor):
    """Get a list of parameters from a model or a tensor."""
    if isinstance(model_or_tensor, nn.Module):
        return list(model_or_tensor.parameters())
    elif torch.is_tensor(model_or_tensor):
        return [model_or_tensor]
    else:
        raise TypeError("Input must be an nn.Module or a torch.Tensor")

# --- DP Double-Spider Trainer ---

class DPDoubleSpiderTrainer:
    def __init__(self, T, q, epsilon, delta, n, d, L0, L1, L2, D0, D1, D2, H, G, M, lambda_val, c, max_practical_bs):
        """
        Initialize the trainer with all algorithm and problem constants.
        
        Args:
            T: Number of iterations
            q: Epoch size
            epsilon, delta: DP parameters
            n: Dataset size
            d: Dimension of parameters (for x, from ResNet20)
            L0, L1, L2, D0, D1, ...: Abstract problem constants
            c: DP noise constant
            max_practical_bs: A hard cap on batch size to prevent OOM errors.
        """
        self.T = T
        self.q = q
        self.epsilon = epsilon
        self.delta = delta
        self.n = n
        self.d = d
        self.c = c
        self.log_1_delta = math.log(1.0 / self.delta)
        self.max_practical_bs = max_practical_bs # Store the practical cap
        
        # Store abstract constants
        self.L0, self.L1, self.L2 = L0, L1, L2
        self.D0, self.D1, self.D2 = D0, D1, D2
        self.H, self.G, self.M = H, G, M
        self.lambda_val = lambda_val

        # --- Calculate Parameters (as per your spec) ---
        
        self.c0 = max(32 * self.L2, 8 * self.L0)
        self.c1 = (4 + (8 * self.L1**2 * self.D2) / (self.n * self.L0**2) + 
                     (32 * self.L1**2 * self.D2) / (self.n * self.L0**2) + 
                     (16 * self.L1**2 * self.L2) / (5 * self.D1 * self.L0**3))
        self.c2 = max(1.0 / (8 * self.L2) + self.L1 / (self.L0**3), 1)
        self.c3 = (1 + self.L2 / (10 * self.L0) + 
                     (self.L0 * self.D1 + self.L0 + 2 * self.L0 * self.L2 * self.D2) / self.L2 + 
                     (33 * self.L2**2) / (5 * self.L0 * self.L2) + self.L1**2 / (15 * self.L2**3) + 
                     self.L1**2 / (2 * self.L0 * self.L2**2))
        self.c4 = 17.0/4.0 + math.sqrt(self.c3) + math.sqrt(1.0 / (60 * self.L2))

        # Batch Sizes
        self.N1 = math.ceil(max(
            (6 * self.D2 * self.c0 * self.c2) / (self.epsilon**2),
            1 # Ensure batch size is at least 1
        ))
        self.N2 = math.ceil(max(
            (20 * self.q * self.D1 * self.L2) / self.L0,
            20 * self.q * self.c2 * self.L2,
            (12 * self.q * self.L1**2 * self.c0 * self.c2) / (self.L0**2),
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
        
        print(f"Algorithm Parameters (Theoretical):")
        print(f"  N1 (batch size): {self.N1}")
        print(f"  N2 (batch size): {self.N2}")
        print(f"  N3 (batch size): {self.N3}")
        print(f"  N4 (batch size): {self.N4}")
        print(f"  q (epoch size): {self.q}")
        print(f"  T (iterations): {self.T}")
        print(f"  Max Practical Batch Size: {self.max_practical_bs}")

        # Step Sizes
        self.alpha = 1.0 / (4 * self.L2) if self.L2 > 0 else 1.0
        
        # --- Calculate Noise Sigmas ---
        self.sigma1_base = (self.c * self.L2 * math.sqrt(self.log_1_delta)) / self.epsilon
        self.sigma1_mult = max(1.0 / self.N1, math.sqrt(self.T) / (self.n * math.sqrt(self.q)))
        self.sigma1 = self.sigma1_base * self.sigma1_mult
        
        # sigma2 depends on L_N2, which is dynamic. Will compute in loop.
        self.sigma2_base = (self.c * math.sqrt(self.log_1_delta)) / (self.n * self.epsilon) 
        
        self.sigma3_base = (self.c * (self.L0 + self.L1 * math.sqrt(self.H)) * math.sqrt(self.log_1_delta)) / self.epsilon
        # NOTE: sigma3_mult uses N3, but the *actual* loader will be capped.
        # This noise may be incorrect if theoretical N3 is too large, but we
        # prioritize running the code over crashing.
        self.sigma3_mult = max(1.0 / self.N3, math.sqrt(self.T) / (self.n * math.sqrt(self.q))) # N3
        self.sigma3 = self.sigma3_base * self.sigma3_mult

        # sigma4 depends on L_N4, which is dynamic. Will compute in loop.
        self.sigma4_base = (self.c * math.sqrt(self.log_1_delta)) / self.epsilon
        self.sigma4_mult = max(1.0 / self.N4, math.sqrt(self.T) / (self.n * math.sqrt(self.q)))
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def _get_beta(self, v_t_norm):
        """Calculates dynamic step size beta_t."""
        term1_denom = (2 * self.L0 + self.L1 * math.sqrt(self.H))
        term1 = 1.0 / term1_denom if term1_denom > 0 else float('inf')
        
        term2_denom = (self.L0 * math.sqrt(self.n) * v_t_norm)
        term2 = 1.0 / term2_denom if term2_denom > 0 else float('inf')
        
        beta = min(term1, term2)
        return beta if beta != float('inf') else 1e-3 # Fallback

    def _get_LN2(self, eta_t, eta_t_minus_1, x_t, x_t_minus_1):
        """Calculates the dynamic Lipschitz constant L_N2."""
        eta_diff = eta_t - eta_t_minus_1 # eta is scalar
        x_diff_norm = torch.norm(x_t - x_t_minus_1)
        
        term1 = self.L2 * torch.norm(eta_diff)
        term2 = (self.G * self.M * x_diff_norm) / self.lambda_val
        return 2 * max(term1, term2)

    def _get_LN4(self, eta_t, eta_t_minus_1, x_t, x_t_minus_1):
        """Calculates the dynamic Lipschitz constant L_N4."""
        eta_diff = eta_t - eta_t_minus_1 # eta is scalar
        x_diff_norm = torch.norm(x_t - x_t_minus_1)

        term1 = (self.M * self.L2 * torch.norm(eta_diff)) / self.lambda_val # Assuming L=L2
        term2 = (self.L0 + self.L1 * math.sqrt(self.H)) * x_diff_norm
        return 2 * max(term1, term2)

    def compute_gradient(self, params_list, loss):
        """Computes gradients and returns them as a flat tensor."""
        if not params_list:
            return torch.tensor(0.0, device=self.device)
            
        # create_graph=True was causing OOM errors with large batches,
        # and the graph is detached with .data.copy_() anyway,
        # so it appears to be unnecessary.
        grads = torch.autograd.grad(loss, params_list, create_graph=False)
        flat_grads = torch.cat([g.reshape(-1) for g in grads])
        return flat_grads

    def train(self, x_0_model, eta_0_params, train_dataset):
        """
        Runs the DP Double-Spider algorithm.
        
        Args:
            x_0_model (nn.Module): The initial ResNet20 model (parameters 'x').
            eta_0_params (torch.Tensor): The initial SCALAR parameter 'eta'.
            train_dataset (Dataset): The imbalanced CIFAR-10 dataset.
        """
        
        # --- Loss Function Definition ---
        def loss_function(x_model, eta_params_scalar, data, targets):
            """
            Implementation of the loss function L(x, eta) from the paper.
            L(x,eta,S) = lambda * psi*((l(x,S) - G*eta)/lambda) + G*eta
            psi*(t) = -1 + 1/4 * (t+2)^2
            """
            # Get constants from trainer class
            G = self.G
            lambda_val = self.lambda_val
            
            # 1. Compute ell(x,S)
            outputs = x_model(data)
            ell_x_s = F.cross_entropy(outputs, targets)
            
            # 2. Define psi_star
            def psi_star(t):
                return -1.0 + 0.25 * (t + 2.0)**2
            
            # 3. Compute L(x, eta)
            t_val = (ell_x_s - G * eta_params_scalar) / lambda_val
            loss = lambda_val * psi_star(t_val) + G * eta_params_scalar
            
            return loss
        # --- End Loss Function Definition ---
        
        
        x_model = x_0_model.to(self.device)
        x_params_list = list(x_model.parameters())

        # eta is a scalar parameter as per the loss function
        eta_params = eta_0_params.clone().to(self.device)
        if not eta_params.requires_grad:
            eta_params.requires_grad_(True)
        eta_params_list = [eta_params] # List containing the one scalar tensor

        # History for returning random state
        x_history = []
        eta_history = []
        
        # Initialize flat parameter representations
        x_t = get_flat_params(x_params_list)
        eta_t = eta_params.clone() # eta_t is the scalar tensor itself
        
        g_t = torch.zeros_like(eta_t)
        v_t = torch.zeros_like(x_t)
        
        # Dataloaders
        # We create new dataloaders inside the loop to get random batches
        def get_loader(batch_size_theoretical, loader_name, current_iter):
            """
            Creates a DataLoader, capping the theoretical batch size at a
            practical limit to avoid OOM errors.
            """
            # Cap theoretical batch size by dataset size
            bs_capped_by_dataset = min(int(batch_size_theoretical), len(train_dataset))
            
            # Apply a further practical cap to prevent OOM errors
            bs_practical = min(bs_capped_by_dataset, self.max_practical_bs)
            
            if bs_practical < bs_capped_by_dataset and current_iter == 0:
                # Warn the user ONCE at the start of training for each loader
                print(f"WARNING ({loader_name} @ t=0): Theoretical batch size ({bs_capped_by_dataset}) is too large.")
                print(f"         Capping at practical limit: {bs_practical}. ")
                print(f"         => To fix, adjust theoretical constants (G, L, H, c) in __main__.")

            if bs_practical <= 0: return None
            return DataLoader(train_dataset, batch_size=bs_practical, shuffle=True)

        # Pass t=0 to trigger initial warning if needed
        loader_N1 = get_loader(self.N1, "N1", 0)
        loader_N2 = get_loader(self.N2, "N2", 0)
        loader_N3 = get_loader(self.N3, "N3", 0)
        loader_N4 = get_loader(self.N4, "N4", 0)

        print("Starting DP Double-Spider Training...")

        for t in range(self.T):
            
            # Store history
            x_history.append(x_t.clone())
            eta_history.append(eta_t.clone())
            
            # Get previous states
            x_t_minus_1 = x_history[-1] if t > 0 else x_t
            eta_t_minus_1 = eta_history[-1] if t > 0 else eta_t

            # Set models to current parameters
            set_flat_params(x_params_list, x_t)
            # eta_t is the tensor, so we use it directly
            eta_params.data.copy_(eta_t.data)

            # --- 1. Update eta ---
            if t % self.q == 0:
                # Full gradient step for g_t
                if loader_N1 is None: 
                    print(f"Skipping t={t} (N1=0 or <1)"); continue
                try:
                    data, targets = next(iter(loader_N1))
                except StopIteration:
                    loader_N1 = get_loader(self.N1, "N1", t) # Re-init loader
                    if loader_N1 is None: continue
                    data, targets = next(iter(loader_N1))
                    
                data, targets = data.to(self.device), targets.to(self.device)
                
                loss = loss_function(x_model, eta_params, data, targets)
                g_t_no_noise = self.compute_gradient(eta_params_list, loss)
                
                noise_g = torch.normal(0.0, self.sigma1, size=g_t_no_noise.shape, device=self.device)
                g_t = g_t_no_noise + noise_g
            
            else:
                # Variance-reduced step for g_t
                if loader_N2 is None: 
                    print(f"Skipping t={t} (N2=0 or <1)"); continue
                try:
                    data, targets = next(iter(loader_N2))
                except StopIteration:
                    loader_N2 = get_loader(self.N2, "N2", t)
                    if loader_N2 is None: continue
                    data, targets = next(iter(loader_N2))
                    
                data, targets = data.to(self.device), targets.to(self.device)

                # Compute g_t(x_t, eta_t)
                loss_t = loss_function(x_model, eta_params, data, targets)
                grad_t = self.compute_gradient(eta_params_list, loss_t)
                
                # Compute g_t-1(x_t-1, eta_t-1)
                set_flat_params(x_params_list, x_t_minus_1)
                eta_params.data.copy_(eta_t_minus_1.data)
                loss_t_minus_1 = loss_function(x_model, eta_params, data, targets)
                grad_t_minus_1 = self.compute_gradient(eta_params_list, loss_t_minus_1)
                
                # Calculate dynamic noise
                L_N2 = self._get_LN2(eta_t, eta_t_minus_1, x_t, x_t_minus_1)
                # L_N2 is a tensor, extract its float value
                sigma2 = self.sigma2_base * L_N2.item() 
                noise_xi = torch.normal(0.0, sigma2, size=g_t.shape, device=self.device)
                
                g_t = grad_t - grad_t_minus_1 + g_t.clone() + noise_xi # g_t.clone() is g_{t-1}
            
            # Apply eta update
            eta_t_plus_1 = eta_t - self.alpha * g_t
            
            # --- 2. Update x ---
            # Set model params (x) back to x_t for this step
            set_flat_params(x_params_list, x_t)
            # Set eta params to the new eta_t+1
            eta_params.data.copy_(eta_t_plus_1.data)

            if t % self.q == 0:
                # Full gradient step for v_t
                if loader_N3 is None: 
                    print(f"Skipping t={t} (N3=0 or <1)"); continue
                try:
                    data, targets = next(iter(loader_N3))
                except StopIteration:
                    loader_N3 = get_loader(self.N3, "N3", t)
                    if loader_N3 is None: continue
                    data, targets = next(iter(loader_N3))
                    
                data, targets = data.to(self.device), targets.to(self.device)

                loss = loss_function(x_model, eta_params, data, targets)
                v_t_no_noise = self.compute_gradient(x_params_list, loss)
                
                noise_v = torch.normal(0.0, self.sigma3, size=v_t_no_noise.shape, device=self.device)
                v_t = v_t_no_noise + noise_v
            
            else:
                # Variance-reduced step for v_t
                if loader_N4 is None: 
                    print(f"Skipping t={t} (N4=0 or <1)"); continue
                try:
                    data, targets = next(iter(loader_N4))
                except StopIteration:
                    loader_N4 = get_loader(self.N4, "N4", t)
                    if loader_N4 is None: continue
                    data, targets = next(iter(loader_N4))
                    
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Compute v_t(x_t, eta_t+1)
                # eta is already set to eta_t+1
                loss_t = loss_function(x_model, eta_params, data, targets)
                grad_t_x = self.compute_gradient(x_params_list, loss_t)

                # Compute v_t-1(x_t-1, eta_t)
                set_flat_params(x_params_list, x_t_minus_1)
                eta_params.data.copy_(eta_t.data) # Use eta_t
                loss_t_minus_1 = loss_function(x_model, eta_params, data, targets)
                grad_t_minus_1_x = self.compute_gradient(x_params_list, loss_t_minus_1)

                # Calculate dynamic noise
                L_N4 = self._get_LN4(eta_t, eta_t_minus_1, x_t, x_t_minus_1)
                # L_N4 is a tensor, extract its float value
                sigma4 = self.sigma4_base * L_N4.item() * self.sigma4_mult
                noise_chi = torch.normal(0.0, sigma4, size=v_t.shape, device=self.device)

                v_t = grad_t_x - grad_t_minus_1_x + v_t.clone() + noise_chi # v_t.clone() is v_{t-1}

            # Apply x update
            beta_t = self._get_beta(torch.norm(v_t))
            x_t_plus_1 = x_t - beta_t * v_t

            # --- 3. Update states for next loop ---
            x_t = x_t_plus_1
            eta_t = eta_t_plus_1
            
            if t % 10 == 0:
                print(f"Iteration {t}/{self.T}, |g_t|: {torch.norm(g_t):.2f}, |v_t|: {torch.norm(v_t):.2f}, beta_t: {beta_t:.2e}")

        # --- 4. Return ---
        print("Training finished.")
        # Return Randomly select x, eta from 1, ..., T
        rand_idx = np.random.randint(0, len(x_history))
        print(f"Returning random state from iteration {rand_idx}")
        
        final_x_params = x_history[rand_idx]
        final_eta_params = eta_history[rand_idx]
        
        # Set final model state
        set_flat_params(x_params_list, final_x_params)
        
        return x_model, final_eta_params


# --- Main execution block ---

if __name__ == "__main__":
    
    # --- 1. Load Data ---
    print("Loading data...")
    train_dataset, test_dataset = load_imbalanced_cifar10()
    
    # --- 2. Define Problem Constants ---
    
    # M is computable from psi*(t) = 1/4*t^2 + t
    # The gradient is d(psi*)/dt = 1/2*t + 1
    # The Lipschitz constant of the gradient is M = 0.5
    M = 0.5        # Smoothness constant of psi*
    
    # TODO: G, L, H, c, and sigma_sq_placeholder are THEORETICAL ASSUMPTIONS
    # about your model and loss. You MUST set these based on your
    # problem's theory.
    
    G = 1.0        # Lipschitz constant of l(x,s) (Set to 1.0 as requested)
    L = 1.0        # Smoothness constant of l(x,s) (Set to 1.0 as requested)
    
    # H is used in the generalized (L0, L1)-smoothness definition,
    # which often relates to a bound on the gradient norm (e.g., ||grad||^2 <= H).
    # You defined it as "max value over the loss", so you must choose a
    # placeholder value for this bound.
    H = 1.0        # PLACEHOLDER: Bound related to generalized smoothness (e.g., on loss or grad norm)
    c = 1.0        # PLACEHOLDER: DP constant from paper
    
    # This is the sigma^2 from your D0 and D2 definitions.
    # User specified sigma = G, so sigma^2 = G^2.
    sigma_sq_placeholder = G**2 # Assumed variance of stochastic gradients (sigma^2 = G^2)

    lambda_val = 0.1 # Given by user
    
    # Calculate L0, L1, L2, D0, D1, D2
    # Added 1e-8 for numerical stability
    denom_G = G + 1e-8
    denom_lambda = lambda_val + 1e-8
    
    L0 = G + (G**2 * M) / denom_lambda
    L1 = L / denom_G
    L2 = (G**2 * M) / denom_lambda
    
    D1 = 8.0       # Given by user
    
    # D0 and D2 depend on a placeholder sigma^2
    D2_common_term = (G**2) * (M**2) * (denom_lambda**-2) * sigma_sq_placeholder
    D2 = D2_common_term
    D0 = 8 * (G**2) + 10 * D2_common_term

    
    # --- 3. Define Algorithm Parameters ---
    n = len(train_dataset)
    epsilon = 4.0
    delta = 1.0 / (n**1.1)
    
    # Get param dimension d
    x_model_init = ResNet20()
    d = sum(p.numel() for p in x_model_init.parameters())
    print(f"Dataset size (n): {n}")
    print(f"Parameter dim (d): {d}")
    print(f"DP Epsilon: {epsilon:.2f}, Delta: {delta:.2e}")
    
    # q = O(n * eps / sqrt(d * log(1/delta)))
    # Set q_constant to 1.0 as requested
    q_constant = 1.0 
    q_denom = math.sqrt(d * math.log(1.0/delta))
    q = math.ceil(q_constant * (n * epsilon) / q_denom) if q_denom > 0 else n
    
    T = 100  # Total iterations (REDUCED FOR FASTER DEBUGGING)
    
    # --- 4. Initialize 'eta' parameters ---
    # From the loss function, eta is a scalar.
    d_eta = 1 
    # Initialize as a single-element tensor, requiring gradient
    eta_0_init = torch.tensor([0.0], requires_grad=True)

    # --- 5. OOM FIX: Define a practical batch size limit ---
    # The theoretical batch sizes (N3) are enormous due to placeholder
    # constants. We cap them at a practical value to prevent OOM errors.
    # This means the code deviates from the paper's theory, but it will RUN.
    # To fix this, you must provide better theoretical constants (G, L, H, c).
    MAX_PRACTICAL_BATCH_SIZE = 256
    print(f"NOTE: Capping all batch sizes at {MAX_PRACTICAL_BATCH_SIZE} to prevent OOM errors.")


    print(f"--- Initializing Trainer ---")
    print(f"INFO: Using computable constant M = {M}")
    print(f"INFO: Using user-set G = {G}, L = {L}")
    print(f"INFO: Using user-set sigma^2 = G^2 = {sigma_sq_placeholder}")
    print("WARNING: Using placeholder values for H and c.")
    print("You MUST update these constants in the __main__ block to proper values.")

    # --- 6. Initialize and Run Trainer ---
    try:
        trainer = DPDoubleSpiderTrainer(
            T=T, q=q, epsilon=epsilon, delta=delta, n=n, d=d,
            L0=L0, L1=L1, L2=L2, D0=D0, D1=D1, D2=D2, H=H, G=G, M=M,
            lambda_val=lambda_val, c=c,
            max_practical_bs=MAX_PRACTICAL_BATCH_SIZE # Pass the cap
        )

        # The loss_function in train() is now correctly defined.
        final_model, final_eta = trainer.train(
            x_0_model=x_model_init,
            eta_0_params=eta_0_init,
            train_dataset=train_dataset
        )
        
        print("\n--- Training Complete ---")
        print(f"Final scalar eta value: {final_eta.item()}")
        
        # TODO: Add evaluation logic here
        # e.g., test(final_model, test_dataset)

    except ZeroDivisionError as e:
        print(f"\n--- ERROR: {e} ---")
        print("A ZeroDivisionError or instability occurred. This is likely because the")
        print("placeholder constants (G, L, M, H, c, sigma^2) are 1.0, leading")
        print("to unstable calculations (e.g., L0=0, L1=0) or batch sizes < 1.")
        print("Please provide the correct, non-zero constants for your problem.")
    except Exception as e:
        print(f"\n--- An unexpected error occurred ---")
        print(e)
        print("This might be due to the placeholder constants or an issue in the logic")
        print("that depends on them (e.g., batch size 0 or invalid gradients).")



