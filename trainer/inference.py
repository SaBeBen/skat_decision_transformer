import numpy as np
import torch

from dt_skat_environment.environment import ACT_DIM


# Function that gets an action from the model using autoregressive prediction
# with a window of the previous 12 timesteps (one game)
@torch.no_grad()
def get_action(model, states, actions, rewards, returns_to_go, timesteps, action_mask=None):
    # This implementation does not condition on past rewards

    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    # The prediction is conditioned on up to 12 previous time-steps
    states = states[:, -model.config.max_ep_len:]
    actions = actions[:, -model.config.max_ep_len:]
    returns_to_go = returns_to_go[:, -model.config.max_ep_len:]
    timesteps = timesteps[:, -model.config.max_ep_len:]

    # pad all tokens to sequence length, this is required if we process batches
    padding = model.config.max_ep_len - states.shape[1]
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padding, model.config.state_dim)), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, ACT_DIM)), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)
    if action_mask is not None:
        action_mask = action_mask.reshape(1, -1, ACT_DIM)[:, -model.config.max_ep_len:]
        action_mask = torch.cat([torch.zeros((1, padding, ACT_DIM)), action_mask.float()], dim=1).to(model.device)

    # perform the prediction on the device of the model (e.g. for online training on a GPU)
    device = model.device
    state_preds, action_preds, return_preds = model.original_forward(
        states=states.to(device),
        actions=actions.to(device),
        rewards=rewards,
        returns_to_go=returns_to_go.to(device),
        timesteps=timesteps.to(device),
        attention_mask=attention_mask.to(device),
        return_dict=False,
        action_mask=action_mask,
    )
    return action_preds[0, -1].cpu()


@torch.no_grad()
def get_action_batch(model, trajectories, timesteps):
    """
    Action probabilities for the current time step of several trajectories in one forward pass.

    :param trajectories: dicts with the lists "states", "actions", "rtg" and "legal" up to (at least) the time step.
    :param timesteps: the current time step of every trajectory.
    :return: array (batch, ACT_DIM) of action probabilities.
    """
    max_len = model.config.max_ep_len
    batch_size = len(trajectories)
    states = np.zeros((batch_size, max_len, model.config.state_dim), dtype=np.float32)
    actions = np.zeros((batch_size, max_len, ACT_DIM), dtype=np.float32)
    returns_to_go = np.zeros((batch_size, max_len, 1), dtype=np.float32)
    steps = np.zeros((batch_size, max_len), dtype=np.int64)
    attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)
    action_mask = np.zeros((batch_size, max_len, ACT_DIM), dtype=np.float32)

    for b, (trajectory, t) in enumerate(zip(trajectories, timesteps)):
        # left padding as in get_action
        length = min(t + 1, max_len)
        first = t + 1 - length
        states[b, -length:] = np.asarray(trajectory["states"][first:t + 1], dtype=np.float32)
        actions[b, -length:] = np.asarray(trajectory["actions"][first:t + 1], dtype=np.float32)
        returns_to_go[b, -length:, 0] = np.asarray(trajectory["rtg"][first:t + 1], dtype=np.float32)
        steps[b, -length:] = np.arange(first, t + 1)
        attention_mask[b, -length:] = 1
        action_mask[b, -length:] = np.asarray(trajectory["legal"][first:t + 1], dtype=np.float32)

    device = model.device
    _, action_preds, _ = model.original_forward(
        states=torch.from_numpy(states).to(device),
        actions=torch.from_numpy(actions).to(device),
        rewards=None,
        returns_to_go=torch.from_numpy(returns_to_go).to(device),
        timesteps=torch.from_numpy(steps).to(device),
        attention_mask=torch.from_numpy(attention_mask).to(device),
        return_dict=False,
        action_mask=torch.from_numpy(action_mask).to(device),
    )
    return action_preds[:, -1].cpu().numpy()
