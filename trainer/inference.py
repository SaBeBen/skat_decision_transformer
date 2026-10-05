import torch

from dt_skat_environment.environment import ACT_DIM


# Function that gets an action from the model using autoregressive prediction
# with a window of the previous 12 timesteps (one game)
@torch.no_grad()
def get_action(model, states, actions, rewards, returns_to_go, timesteps):
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
    )
    return action_preds[0, -1].cpu()
