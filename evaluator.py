import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import copy
from models import ValueNetVBfull, calculate_kl_terms_informative, valueNetlastVB
from bounds import  compute_inf_risk_invkl, compute_risk_rpb
from plotting import NR_plot_loss, plot_loss
import numpy as np 


# Model Training and Run Recursive cases
class PerformanceEval:
    def __init__(self, env, model_type, seed, eval_episodes, test_episodes, env_limit, num_portions, device="cpu"):
        self.env = env
        self.seed = seed
        self.eval_episodes = eval_episodes
        self.test_episodes = test_episodes
        self.env_limit = env_limit
        self.device = device
        self.model_type = model_type
        self.gamma = 0.99
        self.learning_rate = 2e-2
        self.batch_size = 32
        self.epochs = 1
        self.local_reparametrization = True
        self.num_portions = num_portions
        self.emp_risk_data = [[] for _ in range(eval_episodes)]
        self.posteriors = []
        self.loss_history = []
        self.minmax = []
        self.kl_final = None
        self.num_repeats = 1
        self.MCnet = None
        self.kl_NR_I =0
        
       
        # Load data
        self.data = self.get_data()
        self.values, self.test_tuples = self.value_tuples()
        self.G = self.get_G(self.values, mode="train")  # Get G values for training data

        # Loss function & model initialization
        self.loss_fn = torch.nn.MSELoss()
        self.init_value_net()
        self.init_value_net("priornet")
        self.priornet.load_state_dict(self.net.state_dict())

        # Train model
        self.train()

        # Plot loss curve
        plot_loss(self)

        risklist = []
        predictions_mv = []
        prediction = []
        for i in range(self.eval_episodes):
            if len(self.emp_risk_data[i]) > 0:
                risklist.append(self.emp_risk_data[i])
        self.excess, self.kllist, self.E_tlist, self.B_tlist, self.laststep, self.max_B, self.B_T_inv, self.boundsize, self.posterioremploss = compute_risk_rpb(risklist)
        self.risk_MC = 0 

    def load_path(self, i):
        """Loads stored trajectories."""
        path_i = (
            f"Ant_validationround_100000/seed_0{self.seed}/performance_infos_{i}.pt"
        )
        return torch.load(path_i, map_location=torch.device('cpu'), weights_only=False)

    def get_data(self):
        """Load data for all episodes."""
        return [self.load_path(i) for i in range(self.test_episodes)]

    def value_tuples(self):
        """Extract (state, reward) pairs from evaluation and test episodes."""
        values = [[] for _ in range(self.test_episodes)]

        for i, episode in enumerate(self.data[: self.test_episodes]):
            trajectory = episode["trajectory"]
            values[i].extend((step[2], step[5]) for step in trajectory)

        # Split the values into evaluation and test sets
        return values[: self.eval_episodes], values[100 :]

    def get_G(self, values, mode="train"):
        """Compute return G using discounted rewards."""

        G = []
        all_returns = []  # List to store all returns for plotting
        for episode in values:
            episode_G = []
            next_return = 0  # Initialize the return for the next step
            for s, r in reversed(episode):  # Iterate in reverse order
                next_return = r + self.gamma * next_return
                episode_G.append((s, next_return))
            episode_G.reverse()  # Reverse to restore original order

            if any(x in self.env.lower() for x in ["ant", "cheetah"]):
                sampled_episode_G = episode_G[::5]
            elif any(x in self.env.lower() for x in ["humanoid", "hopper", "walker2d"]):
                sampled_episode_G = episode_G[::3] 
            G.append(sampled_episode_G)

            all_returns.extend([t[1] for t in sampled_episode_G])  # collect returns

        return G


    def init_value_net(self, net_name="net"):
        """Initialize a new Value Network based on the provided network name and assign it as an attribute."""
        s, _ = self.G[0][0]  # Extract first state tensor

        # Create the network based on the net_name and assign it as an attribute
        if net_name == "net":
            if self.model_type == "fullvb":
                self.net = ValueNetVBfull((s.shape[0],), self.local_reparametrization).to(self.device)
            elif self.model_type == "lastlayervb":
                self.net = valueNetlastVB((s.shape[0],), self.local_reparametrization).to(self.device)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
                
        elif net_name == "priornet":
            if self.model_type == "fullvb":
                self.priornet = ValueNetVBfull((s.shape[0],), self.local_reparametrization).to(self.device)
            elif self.model_type == "lastlayervb":
                self.priornet = valueNetlastVB((s.shape[0],), self.local_reparametrization).to(self.device)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
     
                
        else:
            raise ValueError(f"Unknown network name: {net_name}")

        if net_name == "net":
            network = self.net
            self.optim_net = optim.Adam(network.parameters(), lr=self.learning_rate)#, weight_decay=1e-4)
            self.scheduler_net = optim.lr_scheduler.StepLR(
            self.optim_net, step_size=10, gamma=0.5
        )
        elif net_name == "priornet":
            network = self.priornet
            self.optim_priornet = optim.Adam(network.parameters(), lr=self.learning_rate)#, weight_decay=1e-4)
            self.scheduler_priornet = optim.lr_scheduler.StepLR(
            self.optim_priornet, step_size=10, gamma=0.5
        )
  

    def excess_loss(self, emploss, emploss_prior, gamma_t = 0.05):
        """Compute the excess loss."""
        return (emploss - gamma_t * emploss_prior) #.clamp(min=0)
    
    def E_t(self, excessloss, kl , n_bound , B , delta = 0.025, mu=0, gamma_t = 0.5):
        """Compute the excess loss."""
        z_sup = (torch.maximum(torch.tensor(0.0), excessloss - mu)).mean()
        z_inf = (torch.maximum(torch.tensor(0.0), mu - excessloss)).mean()
        kl_ratio = torch.div(kl + np.log((4 * np.sqrt(n_bound)) / delta), 2*n_bound,)
        E_t =  mu + (B - mu) * (z_sup / (B - mu) + torch.sqrt(kl_ratio)) - (mu + gamma_t * B) * (z_inf / (mu + gamma_t * B) - torch.sqrt(kl_ratio))
        # E_t = z_sup - z_inf + (3/2 *B * torch.sqrt(kl_ratio))
        return E_t
    
    def get_step_sizes(self, total_episodes):
        ratios = [0.03, 0.07, 0.13, 0.25, 0.5, 1.0]
        steps =[]
        last = 0
        for r in ratios:
            step = max(last + 1, int(total_episodes * r))
            step = min(step, total_episodes)  # Don't exceed total
            steps.append(step)
            last = step
        return steps

    def train(self):
        """Train the value network using progressive episode inclusion."""
        total_episodes = len(self.G)
        half = int(len(self.G)/2)
        if self.num_portions == 6:
            step_sizes = self.get_step_sizes(total_episodes)  # 6 portions
        elif self.num_portions ==2:
            step_sizes = [half, total_episodes]  # 2 portion
        else:
            raise ValueError("num_portions must be either 2 or 6")
        previous_step = 0

        # Precompute episode data for efficiency
        episode_data = [
            (
                torch.stack([s for s, _ in self.G[i]]),
                torch.tensor([t for _, t in self.G[i]], dtype=torch.float32),
            )
            for i in range(total_episodes)
        ]
       
        # Initialize loss tracking
        self.emploss_prior = [[] for _ in range(len(step_sizes))]
        self.emploss_posterior = [[] for _ in range(len(step_sizes))]

        
        self.priornet.eval()  # Ensure priornet is in eval mode
        B = self.env_limit
        self.B_list = []
        self.upper_limit = B
        self.tariningloss = []
        for i, step in enumerate(step_sizes):
            
            episode_indices = range(previous_step, min(step, total_episodes))

            # use for n_bound in E_t
            len_n_bound = 0
            for j in range(previous_step, total_episodes):
                len_n_bound += len(self.G[j])

            # Concatenate states and targets for selected episodes
            all_states = torch.cat([episode_data[i][0] for i in episode_indices])
            all_targets = torch.cat([episode_data[i][1] for i in episode_indices])

            # Create DataLoader
            dataset = TensorDataset(all_states, all_targets)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            max_posterior_loss = torch.tensor(float('-inf'), device=self.device)  # Initialize with a very small value

            # Training loop
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                
                for states, targets in dataloader:
                    states, targets = states.to(self.device), targets.to(self.device)
                    ## Forward pass and loss computation
                    posterior_predictions = self.net(states)[0]
                    
                    posteriorloss = (posterior_predictions - targets.view(-1, 1)) ** 2
                    posteriorloss = torch.clamp(posteriorloss, max=self.upper_limit)
                    max_posterior_loss = torch.max(max_posterior_loss, posteriorloss.max())

                    with torch.no_grad():
                        prior_prediction = self.priornet(states)[0]
                        priorloss = (prior_prediction - targets.view(-1, 1)) ** 2
                        priorloss = torch.clamp(priorloss, max=self.upper_limit)
                    kl = calculate_kl_terms_informative(self.net, self.priornet)[0]

                    # Calculate final loss using excess_loss function
                    excess = self.excess_loss(posteriorloss, priorloss)
                    #excess = torch.clamp(excess, max=self.upper_limit)
                    loss = self.E_t(excess, kl, len_n_bound, self.upper_limit)  #len(all_states), change last input to B  

                    
                    ## Track loss (convert to scalar values)
                    self.emploss_prior[i].append(priorloss.mean().detach().cpu().item())
                    self.emploss_posterior[i].append(posteriorloss.mean().detach().cpu().item())
                    ## Backward pass and optimization
                    self.optim_net.zero_grad()
                    loss.backward()

                    # Clip gradients for stability
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)

                    # Update the network
                    self.optim_net.step()
                    
                    # Track epoch loss
                    epoch_loss += loss.item()

                # Log average loss for the epoch
                avg_epoch_loss = epoch_loss / len(dataloader)
                print(
                    f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_epoch_loss:.4f}, Episodes Considered: {len(episode_indices)}"
                )
                

                self.loss_history.append(avg_epoch_loss)
                self.tariningloss.append(((torch.maximum(torch.tensor(0.0), excess )).mean(),(torch.maximum(torch.tensor(0.0), 0 - excess)).mean(), kl, loss, excess.mean()))
                self.scheduler_net.step()

            # Call risk calculation after training on the current portion of episodes
            self.risk_input(previous_step, self.upper_limit, step)
            self.B_list.append(self.upper_limit)

            # get prior estimate for test data from priornet
            if i==0:
                self.priornet_uninform = copy.deepcopy(self.priornet)

            # get informative kl for NR-I
            if step == total_episodes:
                self.kl_NR_I = calculate_kl_terms_informative(self.net, self.priornet)[0]

            # Update priornet with the current trained net
            previous_step = step
            self.priornet.load_state_dict(self.net.state_dict())
        
       



    def risk_input(self, prev_step, upper_limit, step):
        """Prepare data for risk computation."""

        kl = calculate_kl_terms_informative(self.net, self.priornet)[0]
        length = 0
        for j in range(prev_step, step):
            length += len(self.G[j])
        max_postloss = torch.tensor(float('-inf'), device=self.device)
        min_postloss = torch.tensor(float('inf'), device=self.device)
        max_priorloss = torch.tensor(float('-inf'), device=self.device)
        min_priorloss = torch.tensor(float('inf'), device=self.device)
     
        for tuple_list in self.G[prev_step:]:
            for state, target in tuple_list:
                target = torch.tensor(target, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    prediction, mean_var = self.net(state.to(self.device))
                    prior_prediction = self.priornet(state.to(self.device))[0]
                    emploss = (prediction - target.view(-1, 1)) ** 2
                    max_postloss = torch.max(max_postloss, emploss)
                    min_postloss = torch.min(min_postloss, emploss)
                    emploss_ = torch.clamp(emploss, max=upper_limit) 
                    
        
                    priorloss = (prior_prediction - target.view(-1, 1)) ** 2
                    max_priorloss = torch.max(max_priorloss, priorloss)
                    min_priorloss = torch.min(min_priorloss, priorloss)
                    priorloss_ = torch.clamp(priorloss, max=upper_limit)  
                    excess_loss = self.excess_loss(emploss_, priorloss_, 0.5)

                self.emp_risk_data[prev_step].append(
                        (state, target, prediction, excess_loss, upper_limit, kl, mean_var, emploss_)  #upper_limit
                )
        self.minmax.append((max_postloss, min_postloss, max_priorloss, min_priorloss))

        #plotting excess loss
        plot_loss(self)


    def predict(self):
        """Predict values for test data and compare with ground truth without batch processing."""
        self.G_test = self.get_G(self.test_tuples, mode="test")  # Get G values for test data

        # Flatten and convert data to tensors
        all_states = torch.stack([s for episode in self.G_test for s, _ in episode]).to(self.device)
        all_targets = torch.tensor(
            [t for episode in self.G_test for _, t in episode], dtype=torch.float32
        ).to(self.device)  # Move targets to device early

        all_states_train = torch.stack([s for episode in self.G for s, _ in episode]).to(self.device)
        all_targets_train = torch.tensor(
            [t for episode in self.G for _, t in episode], dtype=torch.float32
        ).to(self.device)  # Move targets to device early
        max_ = self.env_limit
        total_predictions = []
        total_predictions_train = []
     
    
        with torch.no_grad():
            for state in all_states:   
                predictions, _ = self.net(state) 
                total_predictions.append(predictions)
            for state_t in all_states_train:
                predictions_t, _ = self.net(state_t) 
                total_predictions_train.append(predictions_t)

        # Flatten outputs
        predictions_ =  torch.stack(total_predictions).flatten()
        predictions_train_ =  torch.stack(total_predictions_train).flatten() 

        # Compute loss
        loss = torch.clamp((predictions_ - all_targets) ** 2, max=max_).mean()
        loss_train = torch.clamp((predictions_train_ - all_targets_train) ** 2, max=max_).mean()
        
        return loss, loss_train
    






# Model Training and Run Non-Recursive cases
class NR_PerformanceEval:
    def __init__(self, env, model_type, loss_type, seed, eval_episodes, test_episodes, limit, epochs=1, batch_size=32, device="cpu"):
        self.env = env
        self.seed = seed
        self.eval_episodes = eval_episodes
        self.test_episodes = test_episodes
        self.device = torch.device(device)
        self.gamma = 0.99 
        self.learning_rate = 2e-2 
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.model_type = model_type
        self.env_limit = limit

        self.emp_risk_data_informative = [[] for _ in range(eval_episodes)]
        self.posteriors = []
        self.loss_history = []
        self.minmax = []
        self.kl_final = None
        self.local_reparametrization = False

        self.data = self.get_data()
        self.values, self.test_tuples = self.value_tuples()
        self.G = self.get_G(self.values, mode="train")

        self.loss_fn = torch.nn.MSELoss()
        self.init_value_net()
        self.init_value_net("priornet")
        self.priornet.load_state_dict(self.net.state_dict())

        self.train()
        NR_plot_loss(self)

    def calculate_bound(self):
        risklist = [r for r in self.emp_risk_data_informative if r]
        self.NR_results = compute_inf_risk_invkl(risklist[0])
        self.NR_results_loss = self.NR_results["fullloss"]
        self._NR_n_bound = len(risklist[0])

    def load_path(self, i):
        path_i = (
            f"Ant_validationround_100000/seed_0{self.seed}/performance_infos_{i}.pt"
        )
        return torch.load(path_i, map_location=torch.device('cpu'), weights_only=False)

    def get_data(self):
        return [self.load_path(i) for i in range(self.test_episodes)]

    def value_tuples(self):
        values = [[] for _ in range(self.test_episodes)]
        for i, episode in enumerate(self.data[:self.test_episodes]):
            trajectory = episode["trajectory"]
            values[i].extend((step[2], step[5]) for step in trajectory)
        return values[:self.eval_episodes], values[100:]

    def get_G(self, values, mode="train"):
        G = []
        all_returns = []
        for episode in values:
            episode_G = []
            next_return = 0
            for s, r in reversed(episode):
                next_return = r + self.gamma * next_return
                episode_G.append((s, next_return))
            episode_G.reverse()
            if any(x in self.env.lower() for x in ["ant", "cheetah"]):
                sampled_episode_G = episode_G[::5]
            elif any(x in self.env.lower() for x in ["humanoid", "hopper", "walker2d"]):
                sampled_episode_G = episode_G[::3] 
            G.append(sampled_episode_G)
            
            all_returns.extend([t[1] for t in sampled_episode_G])  # collect returns
        return G

    def init_value_net(self, net_name="net"):
        s, _ = self.G[0][0]
        net_cls = ValueNetVBfull
        net_instance = net_cls((s.shape[0],), self.local_reparametrization).to(self.device)
        

        if net_name == "net":
            self.net = net_instance
            self.optim_net = optim.Adam(self.net.parameters(), lr=self.learning_rate)#, weight_decay=1e-4)
            self.scheduler_net = optim.lr_scheduler.StepLR(self.optim_net, step_size=10, gamma=0.5)
        elif net_name == "priornet":
            self.priornet = net_instance
            self.optim_priornet = optim.Adam(self.priornet.parameters(), lr=self.learning_rate)#, weight_decay=1e-4)
            self.scheduler_priornet = optim.lr_scheduler.StepLR(self.optim_priornet, step_size=10, gamma=0.5)

    def non_recursive_loss(self, loss, kl, n_bound, B, delta=0.025):
        kl_ratio = (kl + np.log((2 * np.sqrt(n_bound)) / delta)) / (2 * n_bound)
        return loss + B * torch.sqrt(kl_ratio)

    def train(self):
        total_episodes = len(self.G)
        half = int(len(self.G)/2)
        if self.loss_type == "noninformative":
            step_sizes = [ total_episodes]
        else:
            step_sizes = [half, total_episodes]
        previous_step = 0
        self.emploss_prior = [[] for _ in range(len(step_sizes))]
        self.emploss_posterior = [[] for _ in range(len(step_sizes))]
        episode_data = [
            (
                torch.stack([s for s, _ in self.G[i]]),
                torch.tensor([t for _, t in self.G[i]], dtype=torch.float32),
            )
            for i in range(total_episodes)
        ]

        self.kl_train_list = []
        self.upper_limit = self.env_limit

        for i, step in enumerate(step_sizes):
            episode_indices = range(previous_step, min(step, total_episodes))
            n_bound = sum(len(self.G[j]) for j in range(previous_step, step))

            all_states = torch.cat([episode_data[i][0] for i in episode_indices])
            all_targets = torch.cat([episode_data[i][1] for i in episode_indices])
            dataloader = DataLoader(TensorDataset(all_states, all_targets), batch_size=self.batch_size, shuffle=True)
            

            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for states, targets in dataloader:
                    states, targets = states.to(self.device), targets.to(self.device)
                    posterior_predictions = self.net(states)[0]
                    posteriorloss = torch.clamp((posterior_predictions - targets.view(-1, 1)) ** 2, max=self.upper_limit)

                    with torch.no_grad():
                        prior_prediction = self.priornet(states)[0]
                        priorloss = torch.clamp((prior_prediction - targets.view(-1, 1)) ** 2, max=self.upper_limit)

                    kl = calculate_kl_terms_informative(self.net, self.priornet)[0]
                    self.kl_train_list.append(kl.item())
                    mse_loss = self.loss_fn(posterior_predictions, targets.view(-1, 1))
                    mse_loss = torch.clamp(mse_loss, max=self.upper_limit)
                    loss = self.non_recursive_loss(mse_loss, kl, n_bound, self.upper_limit)

                    self.emploss_prior[i].append(priorloss.mean().detach().cpu().item())
                    self.emploss_posterior[i].append(posteriorloss.mean().detach().cpu().item())

                    self.optim_net.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1)
                    self.optim_net.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}, Episodes: {len(episode_indices)}")
                self.scheduler_net.step()

            if self.loss_type == "informative" and step == half:
                self.priornet.load_state_dict(copy.deepcopy(self.net.state_dict()))

            previous_step = step

        self.risk_input(previous_step, self.upper_limit, step)

    def risk_input(self, prev_step, upper_limit, step):
        if self.loss_type == "informative":
            prev_step = int(len(self.G)/2) #50
        elif self.loss_type == "noninformative":
            prev_step = 0

        kl = calculate_kl_terms_informative(self.net, self.priornet)[0]
        self.kl_final = kl.item()

        max_postloss = torch.tensor(float('-inf'), device=self.device)
        min_postloss = torch.tensor(float('inf'), device=self.device)
        max_priorloss = torch.tensor(float('-inf'), device=self.device)
        min_priorloss = torch.tensor(float('inf'), device=self.device)

        for tuple_list in self.G[prev_step:]:
            for state, target in tuple_list:
                target = torch.tensor(target, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    prediction, mean_var = self.net(state.to(self.device))
                    prior_prediction = self.priornet(state.to(self.device))[0]

                    emploss = (prediction - target.view(-1, 1)) ** 2
                    priorloss = (prior_prediction - target.view(-1, 1)) ** 2

                    max_postloss = torch.max(max_postloss, emploss)
                    min_postloss = torch.min(min_postloss, emploss)
                    max_priorloss = torch.max(max_priorloss, priorloss)
                    min_priorloss = torch.min(min_priorloss, priorloss)

                    emploss_ = torch.clamp(emploss, max=upper_limit)
                    self.emp_risk_data_informative[prev_step].append(
                        (state, target, prediction, upper_limit, kl, mean_var, emploss_)
                    )

        self.minmax.append((max_postloss, min_postloss, max_priorloss, min_priorloss))
    
        # plotting excess loss
        NR_plot_loss(self)

    def predict(self):
        self.G_test = self.get_G(self.test_tuples,"test")

        all_states = torch.stack([s for episode in self.G_test for s, _ in episode]).to(self.device)
        all_targets = torch.tensor([t for episode in self.G_test for _, t in episode], dtype=torch.float32).to(self.device)

        all_states_train = torch.stack([s for episode in self.G for s, _ in episode]).to(self.device)
        all_targets_train = torch.tensor([t for episode in self.G for _, t in episode], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions_test = torch.stack([self.net(state)[0] for state in all_states]).flatten()
            predictions_train = torch.stack([self.net(state)[0] for state in all_states_train]).flatten()

        loss_test = torch.clamp((predictions_test - all_targets) ** 2, max=self.env_limit).mean()
        loss_train = torch.clamp((predictions_train - all_targets_train) ** 2, max=self.env_limit).mean()

        return loss_test, loss_train


    
    