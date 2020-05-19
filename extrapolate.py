
import train_inverse 
import train_forward

from push_env import PushingEnv
from train_forward import plan_forward_model_extrapolate

if __name__ == "__main__":
    # train inverse model
    inverse = train_inverse.Model(4,4)
    inverse.load("models")

    forward = train_forward.Model(6,2)
    forward.load("models")

    env_inverse = PushingEnv(ifRender=False)
    env_forward = PushingEnv(ifRender=False)

    #env_inverse.plan_inverse_model_extrapolate(inverse)
    plan_forward_model_extrapolate(env_forward, forward)