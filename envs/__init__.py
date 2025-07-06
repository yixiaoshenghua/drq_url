from .carla import Carla
from .dmc import DMC, RandomVideoSource
from .dmc_remastered import DMCRemastered
from .metaworld import MetaWorld, ViewMetaWorld, MultiViewMetaWorld
from .wrappers import *
from .robodesk import RoboDesk
# from .od_mujoco import OffDynamicsMujocoEnv
# from .od_envs import *
from .rlbench import RLBench
# from .minecraft import Minecraft
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def make_env(mode, config, llm_packages=None):
        suite, task = config.task.split("_", 1)
        if suite == "dmc":
            env = DMC(
                task, config.action_repeat, (config.render_size, config.render_size), config.dmc_camera)
            env = NormalizeAction(env)
        elif suite == "dmcdriving":
            env = DMC(
                task, config.action_repeat, (config.render_size, config.render_size), config.dmc_camera)
            env = NormalizeAction(env)
            env = RandomVideoSource(env, config.disvideo_dir, (config.render_size, config.render_size), total_frames=1000, grayscale=config.grayscale)
        elif suite == "metaworld":
            task = "-".join(task.split("_"))
            env = MetaWorld(
                task,
                config.seed,
                config.action_repeat,
                (config.render_size, config.render_size),
                config.camera,
            )
            env = NormalizeAction(env)
        elif suite == "mvmetaworld":
            task = "-".join(task.split("_"))
            env = MultiViewMetaWorld(
                task,
                config.seed,
                config.action_repeat,
                (config.render_size, config.render_size),
                config.camera_keys,
            )
            env = NormalizeAction(env)
        elif suite == "viewmetaworld":
            task = "-".join(task.split("_"))
            env = ViewMetaWorld(
                task,
                config.seed,
                config.action_repeat,
                (config.render_size, config.render_size),
                config.camera,
                config.viewpoint_mode,
                config.viewpoint_randomization_type
            )
            env = NormalizeAction(env)
        elif suite == "robodesk":
            env = RoboDesk(
                task,
                reward='dense',# if mode == 'train' else 'success',
                action_repeat = config.action_repeat,
                render_size = (config.render_size, config.render_size),
                time_limit = config.time_limit,
            )
            env = NormalizeAction(env)
            return env
        elif suite == "carla":
            env = Carla(ports=[config.carla_port, config.carla_port + 10],
                             fix_weather=task, frame_skip=config.action_repeat, **config.carla)
            env = NormalizeAction(env)
        elif suite == "dmcr":
            env = DMCRemastered(task, config.action_repeat, (config.render_size, config.render_size),
                                     config.dmc_camera, config.dmcr_vary)
            env = NormalizeAction(env)
        elif suite == "minecraft":
            env = Minecraft(task, config.seed, config.action_repeat, (config.render_size, config.render_size), config.sim_size, config.eval_hard_reset_every if mode == "eval" else config.hard_reset_every)
            env = MultiDiscreteAction(env)
        else:
            raise NotImplementedError(suite)
        # if config.llm_acs_wrap:
        #     env = LLMActionWrapper(env, config, llm_packages, gap_frame=4)
        env = TimeLimit(env, config.time_limit)
        if config.framestack > 1:
            env = FrameStack(env, k=config.framestack)
        return env