class Config():
    env_name = 'gym_onehotgrid-v0'
    gamma = 1.0
    num_trials = 10
    max_ep_len = 50
    replay_mem_size = 2**14

    train_time_seconds = 60*2

    linear_decay = False
    train_in_epochs = True
    # train_in_epochs = False
    if train_in_epochs:
        period_train_in_epochs = 20
        num_target_reset = 2
        num_epochs = 2
        batch_size = 256
        ep_start = 1.0
        ep_end = 0.01
        ep_decay = 5000
    else:
        period_target_reset = 2000
        batch_size = 64
        ep_start = 1.0
        ep_end = 0.01
        ep_decay = 5000
    assert ep_start >= ep_end