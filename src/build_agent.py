from stable_baselines3 import PPO

def build_ppo_agent(env, learning_rate=3e-4):
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=learning_rate,
        batch_size=256,
        n_steps=2048,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )
    return model
