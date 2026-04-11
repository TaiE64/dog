"""ALaM-style dual-policy ActorCritic for 3-legged locomotion + manipulation.

Reference: "Robust Pedipulation on Quadruped Robots via Gravitational-moment Minimization"
           Shin et al., IJCAS 2025.

Architecture (Fig. 2 in paper):
    1. Manipulation actor:  policy_obs + command_obs → [a_man_leg(3), a_man_diy(4), base_cmd(3)]
    2. Privileged encoder:  privileged_obs → z (latent, training only)
    3. Locomotion actor:    policy_obs + command_obs + base_cmd + z → a_loc(12)
    4. Action composition:  mask loco output by leg_state, place manip actions
    5. Shared critic:       all obs → value

The base_cmd output from manipulation actor is fed INTO the locomotion actor,
creating a coordination gradient path (manip tells loco where to walk).
Gradient isolation is achieved via leg_state masking (Sec 3.2.3): the
manipulation leg's entries in a_loc are zeroed, so loco actor only learns
to control the 3 locomotion legs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization


class ALaMActorCritic(nn.Module):
    """Dual-policy actor-critic with sequential manip→loco computation and leg masking."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # Manipulation actor: outputs [a_man_leg(3), a_man_diy(4), base_cmd(3)] = 10
        num_manip_leg_actions: int = 3,
        num_manip_diy_actions: int = 4,
        num_base_cmd: int = 3,
        manip_actor_hidden_dims: list[int] = [256, 128],
        # Locomotion actor: outputs a_loc(12)
        num_loco_actions: int = 12,
        loco_actor_hidden_dims: list[int] = [512, 256, 128],
        # Privileged encoder (training only)
        priv_latent_dim: int = 16,
        priv_encoder_hidden_dims: list[int] = [64, 32],
        # Critic
        critic_hidden_dims: list[int] = [512, 256, 128],
        # Common
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ALaMActorCritic.__init__ got unexpected arguments, "
                "which will be ignored: " + str(list(kwargs.keys()))
            )
        super().__init__()

        # --- Action dimensions ---
        self.num_loco_actions = num_loco_actions  # 12 (all leg joints)
        self.num_manip_leg_actions = num_manip_leg_actions  # 3 (one leg)
        self.num_manip_diy_actions = num_manip_diy_actions  # 4 (diy joints)
        self.num_base_cmd = num_base_cmd  # 3 (vx, vy, wz)
        self.num_manip_output = num_manip_leg_actions + num_manip_diy_actions + num_base_cmd  # 10
        self.priv_latent_dim = priv_latent_dim

        # Standard Go2: 12 leg only. DIY leg: 12 leg + diy_actions * 2 arms.
        if num_manip_diy_actions > 0:
            total_env_actions = num_loco_actions + num_manip_diy_actions * 2
        else:
            total_env_actions = num_loco_actions  # pure pedipulation, no diy
        assert total_env_actions == num_actions, (
            f"Expected {total_env_actions} env actions, got {num_actions}"
        )

        # --- Observation dimensions ---
        self.obs_groups = obs_groups

        # Policy obs: standard proprioceptive + command (goal + leg_state)
        num_policy_obs = 0
        for g in obs_groups["policy"]:
            assert len(obs[g].shape) == 2
            num_policy_obs += obs[g].shape[-1]

        # Critic obs
        num_critic_obs = 0
        for g in obs_groups["critic"]:
            assert len(obs[g].shape) == 2
            num_critic_obs += obs[g].shape[-1]

        # Privileged obs (optional, for encoder)
        self.has_privileged = "privileged" in obs_groups and len(obs_groups["privileged"]) > 0
        num_priv_obs = 0
        if self.has_privileged:
            for g in obs_groups["privileged"]:
                if g in obs:
                    num_priv_obs += obs[g].shape[-1]

        # --- Manipulation Actor ---
        # Input: policy_obs (proprioceptive + command with goal & leg_state)
        # Output: [a_man_leg(3), a_man_diy(4), base_cmd(3)] = 10
        self.manip_actor = MLP(num_policy_obs, self.num_manip_output, manip_actor_hidden_dims, activation)
        print(f"Manipulation Actor MLP: {self.manip_actor}")

        # --- Privileged Encoder (training only) ---
        if self.has_privileged and num_priv_obs > 0:
            self.priv_encoder = MLP(num_priv_obs, priv_latent_dim, priv_encoder_hidden_dims, activation)
            print(f"Privileged Encoder MLP: {self.priv_encoder}")
            loco_extra_input = num_base_cmd + priv_latent_dim
        else:
            self.priv_encoder = None
            loco_extra_input = num_base_cmd

        # --- Locomotion Actor ---
        # Input: policy_obs + base_cmd(3) [+ z(priv_latent_dim)]
        # Output: a_loc(12)
        self.loco_actor = MLP(
            num_policy_obs + loco_extra_input, num_loco_actions,
            loco_actor_hidden_dims, activation,
        )
        print(f"Locomotion Actor MLP: {self.loco_actor}")

        # --- Shared Critic ---
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # --- Observation normalization ---
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_policy_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        # --- Action noise (20 dims for env actions, NOT for base_cmd) ---
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

        # Buffer for base_cmd (populated during act(), no specific device needed
        # as it's overwritten with GPU tensor on first forward pass)
        self._last_base_cmd = None

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    # ---- Properties expected by PPO ----

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    # ---- Observation helpers ----

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate all policy observation groups."""
        obs_list = [obs[g] for g in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate all critic observation groups."""
        obs_list = [obs[g] for g in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def _get_privileged_obs(self, obs: TensorDict) -> torch.Tensor | None:
        """Get privileged observations for the encoder."""
        if not self.has_privileged:
            return None
        obs_list = []
        for g in self.obs_groups["privileged"]:
            if g in obs:
                obs_list.append(obs[g])
        if not obs_list:
            return None
        return torch.cat(obs_list, dim=-1)

    def _extract_leg_state(self, policy_obs: torch.Tensor) -> torch.Tensor:
        """Extract the 4-dim leg_state from the end of the policy observation.

        Convention: the GoalPositionCommand appends [G_d(3), G_B(3), leg_state(4)]
        as the last observation group ("command"). leg_state is the last 4 dims.
        """
        return policy_obs[:, -4:]

    # ---- Core forward pass ----

    def _compose_action_mean(
        self, policy_obs: torch.Tensor, priv_obs: torch.Tensor | None,
    ) -> torch.Tensor:
        """Sequential forward: manip_actor → base_cmd → loco_actor → compose actions.

        Args:
            policy_obs: (N, policy_dim) concatenated policy observations
            priv_obs: (N, priv_dim) or None, privileged observations

        Returns:
            action_mean: (N, 20) composed environment action mean
        """
        N = policy_obs.shape[0]
        device = policy_obs.device

        # Extract leg_state from policy obs (last 4 dims of command group)
        leg_state = self._extract_leg_state(policy_obs)  # (N, 4)

        # --- Step 1: Manipulation actor ---
        manip_out = self.manip_actor(policy_obs)  # (N, num_manip_output)
        a_man_leg = manip_out[:, :self.num_manip_leg_actions]       # (N, 3)
        if self.num_manip_diy_actions > 0:
            a_man_diy = manip_out[:, 3:3 + self.num_manip_diy_actions]  # (N, 4)
        else:
            a_man_diy = None
        base_cmd = manip_out[:, -self.num_base_cmd:]                # (N, 3)

        # --- Step 2: Privileged encoder (training only) ---
        if self.priv_encoder is not None and priv_obs is not None:
            z = self.priv_encoder(priv_obs)  # (N, priv_latent_dim)
            loco_input = torch.cat([policy_obs, base_cmd, z], dim=-1)
        else:
            z_zeros = torch.zeros(N, self.priv_latent_dim, device=device)
            if self.priv_encoder is not None:
                # Training but no priv obs available (shouldn't happen)
                loco_input = torch.cat([policy_obs, base_cmd, z_zeros], dim=-1)
            else:
                loco_input = torch.cat([policy_obs, base_cmd], dim=-1)

        # --- Step 3: Locomotion actor ---
        loco_mean = self.loco_actor(loco_input)  # (N, 12)

        # --- Step 4: Apply leg state mask (gradient isolation, Sec 3.2.3) ---
        # leg_state: (N, 4), 1=loco, 0=manip. Expand to 12 joints (3 per leg).
        leg_mask_12 = leg_state.repeat_interleave(3, dim=1)  # (N, 12)
        loco_masked = loco_mean * leg_mask_12  # zero out manipulation leg

        # --- Step 5: Place manipulation leg actions ---
        # leg_state has 4 values (one per leg), order set by GoalPositionCommand
        # which resolves actual robot joint ordering at runtime.
        # We find which of the first 2 legs (front legs) has leg_state=0.
        # The manipulation leg's 3 joints start at index (leg_idx * 3) in the 12-dim action.
        with torch.no_grad():
            # Find which leg is the manipulator (leg_state == 0)
            # Only front legs (index 0 or 1 among the 4 legs) can be manipulation legs
            leg0_is_manip = (leg_state[:, 0] == 0).float().unsqueeze(1)  # (N, 1)
            leg1_is_manip = (leg_state[:, 1] == 0).float().unsqueeze(1)  # (N, 1)

        # Place a_man_leg into the correct 3 slots within 12 leg joints
        # Leg 0's joints are at [0:3], Leg 1's joints are at [3:6]
        manip_leg_12 = torch.zeros(N, 12, device=device)
        manip_leg_12[:, 0:3] = a_man_leg * leg0_is_manip
        manip_leg_12[:, 3:6] = a_man_leg * leg1_is_manip
        combined_leg = loco_masked + manip_leg_12   # (N, 12)

        # --- Step 6: Place manipulation DIY actions (if any) ---
        if self.num_manip_diy_actions > 0:
            # a_man_diy = [j1, j2, j3, j4] for the selected leg
            # DIY action layout: [leg0_j1, leg1_j1] then [leg0_j234, leg1_j234]
            # (same order as leg joints — determined by robot, not hardcoded)
            diy_j1 = torch.zeros(N, 2, device=device)
            diy_j1[:, 0] = a_man_diy[:, 0] * leg0_is_manip.squeeze(1)
            diy_j1[:, 1] = a_man_diy[:, 0] * leg1_is_manip.squeeze(1)

            diy_j234 = torch.zeros(N, 6, device=device)
            diy_j234[:, 0:3] = a_man_diy[:, 1:4] * leg0_is_manip
            diy_j234[:, 3:6] = a_man_diy[:, 1:4] * leg1_is_manip

            action_mean = torch.cat([combined_leg, diy_j1, diy_j234], dim=-1)
        else:
            # Standard Go2 pedipulation: 12 leg actions only, EE = foot
            action_mean = combined_leg

        # Store base_cmd for reward function access (detached, no grad needed for reward)
        self._last_base_cmd = base_cmd.detach()

        return action_mean

    def _update_distribution(self, action_mean: torch.Tensor) -> None:
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(action_mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(action_mean)
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")
        self.distribution = Normal(action_mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        policy_obs = self.get_actor_obs(obs)
        policy_obs = self.actor_obs_normalizer(policy_obs)
        priv_obs = self._get_privileged_obs(obs)
        action_mean = self._compose_action_mean(policy_obs, priv_obs)
        self._update_distribution(action_mean)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        policy_obs = self.get_actor_obs(obs)
        policy_obs = self.actor_obs_normalizer(policy_obs)
        # No privileged info at inference
        action_mean = self._compose_action_mean(policy_obs, priv_obs=None)
        return action_mean

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
