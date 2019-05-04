import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_epsilon(steps_done, opts):
    return opts.epsilon_end + (opts.epsilon_start - opts.epsilon_end) * \
        math.exp(-1. * steps_done / opts.epsilon_decay)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DDQN(nn.Module):
    def __init__(self, num_actions, device, memory, gamma=0.999):
        super(DDQN, self).__init__()
        self.conv_image = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(), Flatten()
        )

        self.conv_fft = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(), Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * 12 * 12 * 64, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        self.num_actions = num_actions
        self.optimizer = optim.Adam(self.parameters(), lr=6.25e-5)
        self.memory = memory
        self.gamma = gamma
        self.device = device

    def forward(self, x):
        reconstructions = x[:, :2, :, :]
        masked_rffts = x[:, 2:, :, :]

        image_encoding = self.conv_image(reconstructions)
        rfft_encoding = self.conv_image(masked_rffts)

        return self.fc(torch.cat((image_encoding, rfft_encoding), dim=1))

    def get_action(self, observation, eps_threshold):
        sample = random.random()
        if sample < eps_threshold:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            q_values = self(torch.from_numpy(observation).unsqueeze(0).to(self.device))
        return torch.argmax(q_values, dim=1).item()

    def add_experience(self, observation, action, next_observation, reward):
        self.memory.push(observation, action, next_observation, reward)

    def update_parameters(self, target_net, batch_size):
        if len(self.memory) < batch_size:
            return None, None
        batch = self.memory.sample(batch_size)
        observations = torch.tensor(batch.observation, device=self.device, dtype=torch.float32)
        actions = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)

        not_done_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_observation)),
                                     device=self.device,
                                     dtype=torch.uint8)

        # Compute Q-values and get best action according to online network
        all_q_values = self.forward(observations)
        q_values = all_q_values.gather(1, actions)

        # Compute target values using the best action found
        target_values = torch.zeros(observations.shape[0], device=self.device)
        if not_done_mask.any().item() != 0:
            best_actions = all_q_values.detach()[not_done_mask].max(1)[1]
            not_done_next_observations = torch.cat(
                [torch.tensor([s]) for s in batch.next_observation if s is not None]).to(self.device)
            target_values[not_done_mask] = target_net(not_done_next_observations)\
                .gather(1, best_actions.unsqueeze(1)).squeeze().detach()

        target_values = self.gamma * target_values + rewards

        # loss = F.mse_loss(q_values, target_values.unsqueeze(1))
        loss = F.smooth_l1_loss(q_values, target_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        # Compute total gradient norm (for logging purposes) and then clip gradients
        grad_norm = 0
        for p in list(filter(lambda p: p.grad is not None, self.parameters())):
            grad_norm += (p.grad.data.norm(2).item() ** 2)
        grad_norm = grad_norm ** .5
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 100)
        torch.nn.utils.clip_grad_value_(self.parameters(), 1)

        self.optimizer.step()

        return loss, grad_norm
