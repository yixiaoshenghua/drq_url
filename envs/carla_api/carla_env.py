#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Modified for DBC paper.

# TODO: check & clean

import random
import glob
import os
import sys
import time
from PIL import Image
from PIL.PngImagePlugin import PngInfo

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import math

from dotmap import DotMap

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

        self.start()

    def start(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun(object):
    def __init__(self, azimuth, altitude, random_init=False):
        if random_init:
            self.azimuth = random.uniform(0.0, 360.0)
            self.altitude = random.uniform(0, 45)
            self._t = random.uniform(0.0, 2.0 * math.pi)
        else:
            self.azimuth = azimuth
            self.altitude = altitude
            self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        # according to carla/LibCarla/source/carla/rpc/WeatherParameters.cpp
        # Noon: altitude = 45
        # Sunset: altitude = 15
        # Midnight: altitude = -90
        # for visibility, we set the maximum altitude to 45 and minimum to 0
        min_alt, max_alt = 0, 45
        self.altitude = 0.5 * (max_alt + min_alt) + 0.5 * (max_alt - min_alt) * math.cos(self._t)

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self, precipitation, random_init=False):
        if random_init:
            precipitation = random.uniform(-250.0, 100.0)
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 60.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


WEATHERS = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
    "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "CloudySunset": carla.WeatherParameters.CloudySunset,
    "WetSunset": carla.WeatherParameters.WetSunset,
    "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
    "MidRainSunset": carla.WeatherParameters.MidRainSunset,
    "HardRainSunset": carla.WeatherParameters.HardRainSunset,
    "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
}


class Weather(object):
    def __init__(self, world, changing_weather_speed, fix_weather):
        self.world = world
        self.fix_weather = fix_weather
        if self.fix_weather is None:
            self.reset_to_default()
            self.weather = world.get_weather()
            self.changing_weather_speed = changing_weather_speed
            self._sun = Sun(self.weather.sun_azimuth_angle, self.weather.sun_altitude_angle)
            self._storm = Storm(self.weather.precipitation)
        else:
            # weather is fixed and changing_weather_speed is ignored
            self.reset_weather(fix_weather)
            self.weather = world.get_weather()

    def reset_weather(self, weather):
        assert weather in WEATHERS.keys()
        self.world.set_weather(WEATHERS[weather])

    def reset_to_default(self):
        weather_params = carla.WeatherParameters(sun_altitude_angle=45.)
        self.world.set_weather(weather_params)

    def reset(self):
        if self.fix_weather is None:
            self.reset_to_default()
            self._sun = Sun(self.weather.sun_azimuth_angle, self.weather.sun_altitude_angle, random_init=True)
            self._storm = Storm(self.weather.precipitation, random_init=True)
            self.tick()

    def tick(self):
        if self.fix_weather is None:
            self._sun.tick(self.changing_weather_speed)
            self._storm.tick(self.changing_weather_speed)
            self.weather.cloudiness = self._storm.clouds
            self.weather.precipitation = self._storm.rain
            self.weather.precipitation_deposits = self._storm.puddles
            self.weather.wind_intensity = self._storm.wind
            self.weather.fog_density = self._storm.fog
            self.weather.wetness = self._storm.wetness
            self.weather.sun_azimuth_angle = self._sun.azimuth
            self.weather.sun_altitude_angle = self._sun.altitude
            self.world.set_weather(self.weather)

    def __str__(self):
        if self.fix_weather is None:
            return '%s %s' % (self._sun, self._storm)
        else:
            return self.fix_weather


class CarlaEnv(object):
    def __init__(
        self,
        map_name="Town04",
        render_display=0,  # 0, 1
        record_display_images=0,  # 0, 1
        record_rl_images=0,  # 0, 1
        changing_weather_speed=0.0,  # [0, +inf)
        display_text=0,  # 0, 1
        rl_image_size=84,
        max_episode_steps=1000,
        frame_skip=1,
        is_other_cars=True,
        fov=60,  # degrees for rl camera
        num_cameras=3,
        port=2000,
        fix_weather=None,
        num_other_vehicles=20,
        trafficmanager_port=8000,
        collision_cost=1e-4,
        global_speed_diff=10.0,
        centering_reward_type="div",
        centering_reward_weight=1.0,
        clip_collision_reward=0,
        steer_coeff=1.0,
        centering_border=1.0,
        use_branch_lane_cut=False,
    ):
        if record_display_images:
            assert render_display
        self.map_name = map_name
        self.render_display = render_display
        self.save_display_images = record_display_images
        self.save_rl_images = record_rl_images
        self.changing_weather_speed = changing_weather_speed
        self.display_text = display_text
        self.rl_image_size = rl_image_size
        self._max_episode_steps = max_episode_steps  # DMC uses this
        self.frame_skip = frame_skip
        self.is_other_cars = is_other_cars
        self.num_cameras = num_cameras
        self.fix_weather = fix_weather
        self.num_other_vehicles = num_other_vehicles
        self.trafficmanager_port = trafficmanager_port
        self.collision_cost = collision_cost
        self.global_speed_diff = global_speed_diff
        self.centering_reward_type = centering_reward_type
        self.centering_reward_weight = centering_reward_weight
        self.clip_collision_reward = clip_collision_reward
        self.steer_coeff = steer_coeff
        self.centering_border = centering_border
        self.use_branch_lane_cut = use_branch_lane_cut

        self.actor_list = []

        if self.render_display:
            pygame.init()
            self.display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.font = get_font()
            self.clock = pygame.time.Clock()

        self.client = carla.Client('localhost', port)
        self.client.set_timeout(30.0)
        self.world = self.client.load_world(self.map_name)
        self.map = self.world.get_map()
        assert self.map.name == self.map_name

        # remove old vehicles and sensors (in case they survived)
        self.world.tick()
        actor_list = self.world.get_actors()
        for vehicle in actor_list.filter("*vehicle*"):
            print("Warning: removing old vehicle")
            vehicle.destroy()
        for sensor in actor_list.filter("*sensor*"):
            print("Warning: removing old sensor")
            sensor.destroy()

        self.vehicle = None
        self.vehicle_start_pose = None
        self.vehicles_list = []  # their ids
        self.vehicles = None
        self.reset_vehicle()  # creates self.vehicle
        self.actor_list.append(self.vehicle)

        blueprint_library = self.world.get_blueprint_library()

        if render_display:
            bp = blueprint_library.find('sensor.camera.rgb')
            bp.set_attribute('enable_postprocess_effects', str(True))
            self.camera_rgb = self.world.spawn_actor(bp, carla.Transform(
                carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)), attach_to=self.vehicle)
            self.actor_list.append(self.camera_rgb)

        # we can use up to five cameras, which we'll stitch together
        bp = blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.rl_image_size))
        bp.set_attribute('image_size_y', str(self.rl_image_size))
        bp.set_attribute('fov', str(fov))
        bp.set_attribute('enable_postprocess_effects', str(True))
        location = carla.Location(x=1.6, z=1.7)
        self.camera_rl = self.world.spawn_actor(bp, carla.Transform(
            location, carla.Rotation(yaw=0.0)), attach_to=self.vehicle)
        self.camera_rl_left = self.world.spawn_actor(bp, carla.Transform(
            location, carla.Rotation(yaw=-float(fov))), attach_to=self.vehicle)
        self.camera_rl_lefter = self.world.spawn_actor(bp, carla.Transform(
            location, carla.Rotation(yaw=-2 * float(fov))), attach_to=self.vehicle)
        self.camera_rl_right = self.world.spawn_actor(bp, carla.Transform(
            location, carla.Rotation(yaw=float(fov))), attach_to=self.vehicle)
        self.camera_rl_righter = self.world.spawn_actor(bp, carla.Transform(
            location, carla.Rotation(yaw=2 * float(fov))), attach_to=self.vehicle)
        self.actor_list.append(self.camera_rl)
        self.actor_list.append(self.camera_rl_left)
        self.actor_list.append(self.camera_rl_lefter)
        self.actor_list.append(self.camera_rl_right)
        self.actor_list.append(self.camera_rl_righter)

        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        self.actor_list.append(self.collision_sensor)
        self._collision_intensities_during_last_time_step = []

        if self.save_display_images or self.save_rl_images:
            import datetime
            now = datetime.datetime.now()
            image_dir = "images-" + now.strftime("%Y-%m-%d-%H-%M-%S")
            os.mkdir(image_dir)
            self.image_dir = image_dir

        if self.render_display:
            self.sync_mode = CarlaSyncMode(self.world, self.camera_rgb, self.camera_rl, self.camera_rl_left,
                                           self.camera_rl_lefter, self.camera_rl_right, self.camera_rl_righter, fps=20)
        else:
            self.sync_mode = CarlaSyncMode(self.world, self.camera_rl, self.camera_rl_left,
                                           self.camera_rl_lefter, self.camera_rl_right, self.camera_rl_righter, fps=20)

        # weather
        self.weather = Weather(self.world, self.changing_weather_speed, self.fix_weather)

        # dummy variables given bisim's assumption on deep-mind-control suite APIs
        low = -1.0
        high = 1.0
        self.action_space = DotMap()
        self.action_space.low.min = lambda: low
        self.action_space.high.max = lambda: high
        self.action_space.shape = [2]
        self.observation_space = DotMap()
        self.observation_space.shape = (3, rl_image_size, num_cameras * rl_image_size)
        self.observation_space.dtype = np.dtype(np.uint8)
        self.reward_range = None
        self.metadata = None
        self.action_space.sample = lambda: np.random.uniform(
            low=low, high=high, size=self.action_space.shape[0]).astype(np.float32)

        # roaming carla agent
        self.agent = None
        self.count = 0
        self.dist_s = 0
        self.return_ = 0
        self.collide_count = 0
        self.steer_reward = 0
        self.collision_reward = 0
        self.centering_reward = 0
        self.velocities = []
        self.world.tick()
        self.reset()  # creates self.agent

    def dist_from_center_lane(self, vehicle):
        # assume on highway
        vehicle_location = vehicle.get_location()
        vehicle_waypoint = self.map.get_waypoint(vehicle_location)
        vehicle_xy = np.array([vehicle_location.x, vehicle_location.y])
        vehicle_s = vehicle_waypoint.s
        vehicle_velocity = vehicle.get_velocity()  # Vecor3D
        vehicle_velocity_xy = np.array([vehicle_velocity.x, vehicle_velocity.y])
        speed = np.linalg.norm(vehicle_velocity_xy)

        vehicle_waypoint_closest_to_road = \
            self.map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        road_id = vehicle_waypoint_closest_to_road.road_id
        assert road_id is not None
        lane_id = int(vehicle_waypoint_closest_to_road.lane_id)

        # if vehicle is on shoulder, reposition the road center to the nearest driving lane
        if road_id in [45, 46, 47, 48, 49, 50]:
            if lane_id == 1:
                lane_id = -1
            elif lane_id == -5:
                lane_id = -4
        elif road_id == [102, 886]:
            if lane_id == 1:
                # reposition to 103 and 887, respectively
                road_id += 1
                lane_id = -1
        elif road_id in [660, 559, 718]:
            if lane_id == -1:
                # reposition to 659, 558 and 717, respectively
                road_id -= 1
                lane_id = 1

        if self.use_branch_lane_cut:
            if road_id in [93, 94] or (road_id == 103 and lane_id == -4):
                # on a branch, consider the branch as a shoulder
                road_id = 103
                lane_id = -3

        goal_lane_id = lane_id

        goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s)
        if goal_waypoint is None:
            # try to fix, bit of a hack, with CARLA waypoint discretizations
            carla_waypoint_discretization = 0.02  # meters
            goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s - carla_waypoint_discretization)
            if goal_waypoint is None:
                goal_waypoint = self.map.get_waypoint_xodr(
                    road_id, goal_lane_id, vehicle_s + carla_waypoint_discretization)

        if goal_waypoint is None:
            print("Episode fail: goal waypoint is off the road! (frame %d)" % self.count)
            done, dist, vel_s = True, 100., 0.
        else:
            goal_location = goal_waypoint.transform.location
            goal_xy = np.array([goal_location.x, goal_location.y])
            dist = np.linalg.norm(vehicle_xy - goal_xy)

            next_goal_waypoint = goal_waypoint.next(0.1)  # waypoints are ever 0.02 meters
            if len(next_goal_waypoint) != 1:
                print('warning: {} waypoints (not 1)'.format(len(next_goal_waypoint)))
            if len(next_goal_waypoint) == 0:
                print("Episode done: no more waypoints left. (frame %d)" % self.count)
                done, vel_s = True, 0.
            else:
                location_ahead = next_goal_waypoint[0].transform.location
                highway_vector = np.array([location_ahead.x, location_ahead.y]) - goal_xy
                highway_unit_vector = np.array(highway_vector) / np.linalg.norm(highway_vector)
                vel_s = np.dot(vehicle_velocity_xy, highway_unit_vector)
                done = False

        if road_id in [22, 34, 7, 15]:
            print("Episode fail: deviation from the planned route! (frame {})".format(self.count))
            done = True

        # not algorithm's fault, but the simulator sometimes throws the car in the air wierdly
        if vehicle_velocity.z > 1. and self.count < 20:
            print("Episode done: vertical velocity too high ({}), usually a simulator glitch (frame {})".format(
                vehicle_velocity.z, self.count))
            done = True
        if vehicle_location.z > 0.5 and self.count < 20:
            print("Episode done: vertical velocity too high ({}), usually a simulator glitch (frame {})".format(
                vehicle_location.z, self.count))
            done = True

        return dist, vel_s, speed, done

    def _on_collision(self, event):
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        print('Collision (intensity {})'.format(intensity))
        self._collision_intensities_during_last_time_step.append(intensity)

    def reset(self):
        self.weather.reset()
        self.world.tick()
        self.reset_vehicle()
        self.world.tick()
        self.reset_other_vehicles()
        self.world.tick()
        self.agent = RoamingAgentModified(self.vehicle, follow_traffic_lights=False)
        self.count = 0
        self.dist_s = 0
        self.return_ = 0
        self.collision_reward = 0
        self.steer_reward = 0
        self.centering_reward = 0
        self.velocities = []

        obs, _, _, _ = self.step(action=None)
        return obs

    def reset_vehicle(self):
        start_lane = random.choice([-1, -2, -3, -4])
        self.vehicle_start_pose = self.map.get_waypoint_xodr(road_id=45, lane_id=start_lane, s=100.).transform
        if self.vehicle is None:
            # create vehicle
            blueprint_library = self.world.get_blueprint_library()
            vehicle_blueprint = blueprint_library.find('vehicle.audi.a2')
            self.vehicle = self.world.spawn_actor(vehicle_blueprint, self.vehicle_start_pose)
        else:
            self.vehicle.set_transform(self.vehicle_start_pose)
        self.vehicle.set_target_velocity(carla.Vector3D())
        self.vehicle.set_target_angular_velocity(carla.Vector3D())

    def get_other_vehicles_waypoint(self):
        # Map data from CARLA_0.9.11/CarlaUE4/Content/Carla/Maps/OpenDrive/Town04.xodr
        num_roads = 7
        road_ids = [45, 103, 46, 659, 47, 558, 48]
        road_starts = [200., 0., 0., 0., 0., 0., 0.]
        road_ends = [583., 50., 176., 58., 117., 34., 81.]
        lane_ids = [[-1, -2, -3, -4], [-1, -2, -3, -4], [-1, -2, -3, -4],
                    [1, 2, 3, 4], [-1, -2, -3, -4], [1, 2, 3, 4], [-1, -2, -3, -4]]

        s_range = sum([t - s for s, t in zip(road_starts, road_ends)])  # total 999.
        vehicle_s = np.random.uniform(0., s_range)
        for i in range(num_roads):
            if vehicle_s <= road_ends[i] - road_starts[i]:
                lane_id = random.choice(lane_ids[i])
                return self.map.get_waypoint_xodr(road_ids[i], lane_id, vehicle_s + road_starts[i])
            else:
                vehicle_s -= road_ends[i] - road_starts[i]

        return None

    def reset_other_vehicles(self):
        if not self.is_other_cars:
            return

        # clear out old vehicles
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        self.world.tick()
        self.vehicles_list = []

        # register traffic manager on given port
        traffic_manager = self.client.get_trafficmanager(port=self.trafficmanager_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        traffic_manager.set_synchronous_mode(True)
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        other_car_waypoints = [
            self.get_other_vehicles_waypoint() for _ in range(self.num_other_vehicles)
        ]

        # Spawn vehicles
        batch = []
        for n, waypoint in enumerate(other_car_waypoints):
            transform = waypoint.transform
            transform.location.z += 0.1
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, self.trafficmanager_port)))
        for response in self.client.apply_batch_sync(batch, False):
            self.vehicles_list.append(response.actor_id)

        for response in self.client.apply_batch_sync(batch):
            if response.error:
                pass
            else:
                self.vehicles_list.append(response.actor_id)

        traffic_manager.global_percentage_speed_difference(self.global_speed_diff)

    def compute_steer_action(self):
        control = self.agent.run_step()  # PID decides control.steer
        steer = control.steer
        throttle = control.throttle
        brake = control.brake
        throttle_brake = -brake
        if throttle > 0.:
            throttle_brake = throttle
        steer_action = np.array([steer, throttle_brake], dtype=np.float32)
        return steer_action

    def step(self, action):
        rewards = []
        for _ in range(self.frame_skip):  # default 1
            next_obs, reward, done, info = self._simulator_step(action)
            rewards.append(reward)
            if done:
                break
        return next_obs, np.mean(rewards), done, info

    def _simulator_step(self, action, dt=0.05):

        if self.render_display:
            if should_quit():
                return
            self.clock.tick()

        if action is not None:
            steer = float(action[0])
            throttle_brake = float(action[1])
            if throttle_brake >= 0.0:
                throttle = throttle_brake
                brake = 0.0
            else:
                throttle = 0.0
                brake = -throttle_brake

            assert 0.0 <= throttle <= 1.0
            assert -1.0 <= steer <= 1.0
            assert 0.0 <= brake <= 1.0
            vehicle_control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False
            )
            self.vehicle.apply_control(vehicle_control)
        else:
            throttle, steer, brake = 0., 0., 0.

        # Advance the simulation and wait for the data.
        if self.render_display:
            snapshot, image_rgb, image_rl, image_rl_left, image_rl_lefter, image_rl_right, image_rl_righter = self.sync_mode.tick(
                timeout=2.0)
        else:
            snapshot, image_rl, image_rl_left, image_rl_lefter, image_rl_right, image_rl_righter = self.sync_mode.tick(
                timeout=2.0)

        dist_from_center, vel_s, speed, done = self.dist_from_center_lane(self.vehicle)
        collision_intensities_during_last_time_step = sum(self._collision_intensities_during_last_time_step)
        self._collision_intensities_during_last_time_step.clear()  # clear it ready for next time step
        assert collision_intensities_during_last_time_step >= 0.

        collision_cost = self.collision_cost * collision_intensities_during_last_time_step
        if self.centering_border == 1:
            centering_cost = max(0., dist_from_center - self.centering_border)
        elif self.centering_border == 1.75:
            centering_cost = max(0., dist_from_center - self.centering_border) * 1.5
        else:
            raise NotImplementedError

        if self.clip_collision_reward != 0:
            collision_cost = min(collision_cost, self.clip_collision_reward)
        if self.centering_reward_type == 'div':
            reward = vel_s * dt / (1. + centering_cost) \
                - collision_cost - abs(steer) * self.steer_coeff  # encourages to follow the lane
        elif self.centering_reward_type == 'add':
            reward = vel_s * dt - collision_cost - abs(steer) * self.steer_coeff \
                - self.centering_reward_weight * centering_cost
        else:
            raise NotImplementedError
        self.collision_reward += -collision_cost
        self.steer_reward += -abs(steer) * self.steer_coeff
        self.centering_reward += -self.centering_reward_weight * centering_cost

        self.dist_s += vel_s * dt
        self.return_ += reward

        if self.fix_weather is None:
            self.weather.tick()

        # Draw the display.
        if self.render_display:
            draw_image(self.display, image_rgb)
            if self.display_text:
                self.display.blit(self.font.render('frame %d' % self.count, True, (255, 255, 255)), (8, 10))
                self.display.blit(self.font.render('highway progression %4.1f m/s (%5.2f m) (%5.2f speed)' %
                                  (vel_s, self.dist_s, speed), True, (255, 255, 255)), (8, 28))
                self.display.blit(self.font.render('%5.2f meters off center' %
                                  dist_from_center, True, (255, 255, 255)), (8, 46))
                self.display.blit(self.font.render('%5.2f reward (return %.1f)' %
                                  (reward, self.return_), True, (255, 255, 255)), (8, 64))
                self.display.blit(self.font.render('%5.2f collision intensity ' %
                                  collision_intensities_during_last_time_step, True, (255, 255, 255)), (8, 82))
                self.display.blit(self.font.render('%5.2f thottle, %5.2f steer, %5.2f brake' %
                                  (throttle, steer, brake), True, (255, 255, 255)), (8, 100))
                self.display.blit(self.font.render(str(self.weather), True, (255, 255, 255)), (8, 118))
            pygame.display.flip()

        rgbs = []
        if self.num_cameras == 1:
            ims = [image_rl]
        elif self.num_cameras == 3:
            ims = [image_rl_left, image_rl, image_rl_right]
        elif self.num_cameras == 5:
            ims = [image_rl_lefter, image_rl_left, image_rl, image_rl_right, image_rl_righter]
        else:
            raise ValueError("num cameras must be 1 or 3 or 5")
        for im in ims:
            bgra = np.array(im.raw_data).reshape(self.rl_image_size, self.rl_image_size, 4)  # BGRA format
            bgr = bgra[:, :, :3]  # BGR format (84 x 84 x 3)
            rgb = np.flip(bgr, axis=2)  # RGB format (84 x 84 x 3)
            rgbs.append(rgb)
        rgb = np.concatenate(rgbs, axis=1)  # (84 x 252 x 3)

        # Rowan added
        if self.render_display and self.save_display_images:
            image_name = os.path.join(self.image_dir, "display%08d.jpg" % self.count)
            pygame.image.save(self.display, image_name)
            # ffmpeg -r 20 -pattern_type glob -i 'display*.jpg' carla.mp4
        if self.save_rl_images:
            image_name = os.path.join(self.image_dir, "rl%08d.png" % self.count)
            im = Image.fromarray(rgb)
            metadata = PngInfo()
            metadata.add_text("throttle", str(throttle))
            metadata.add_text("steer", str(steer))
            metadata.add_text("brake", str(brake))
            im.save(image_name, "PNG", pnginfo=metadata)

            # # Example usage:
            # from PIL.PngImagePlugin import PngImageFile
            # im = PngImageFile("rl00001234.png")
            # # Actions are stored in the image's metadata:
            # print("Actions: %s" % im.text)
            # throttle = float(im.text['throttle'])  # range [0, 1]
            # steer = float(im.text['steer'])  # range [-1, 1]
            # brake = float(im.text['brake'])  # range [0, 1]
        self.count += 1

        next_obs = rgb  # (84 x 252 x 3) or (84 x 420 x 3)
        next_obs = np.transpose(next_obs, [2, 0, 1])  # 3 x 84 x 84/252/420
        assert next_obs.shape == self.observation_space.shape
        if self.count >= self._max_episode_steps:
            print("Episode success: I've reached the episode horizon ({}).".format(self._max_episode_steps))
            done = True
        if speed < 0.02 and self.count >= 100 and self.count % 100 == 0:  # a hack, instead of a counter
            print("Episode fail: speed too small ({}), think I'm stuck! (frame {})".format(speed, self.count))
            done = True
        info = None
        return next_obs, reward, done, info

    def finish(self):
        print('destroying actors.')
        for actor in self.actor_list:
            actor.destroy()
        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        time.sleep(0.5)
        pygame.quit()
        print('done.')


class LocalPlannerModified(LocalPlanner):

    def __del__(self):
        pass  # otherwise it deletes our vehicle object

    def run_step(self):
        return super().run_step(debug=False)  # otherwise by default shows waypoints, that interfere with our camera


class RoamingAgentModified(Agent):
    """
    RoamingAgent implements a basic agent that navigates scenes making random
    choices when facing an intersection.

    This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, follow_traffic_lights=True):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(RoamingAgentModified, self).__init__(vehicle)
        self._proximity_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        self._follow_traffic_lights = follow_traffic_lights

        # for throttle 0.5, 0.75, 1.0
        args_lateral_dict = {
            'K_P': 1.0,
            'K_D': 0.005,
            'K_I': 0.0,
            'dt': 1.0 / 20.0}
        opt_dict = {'lateral_control_dict': args_lateral_dict}

        self._local_planner = LocalPlannerModified(self._vehicle, opt_dict)

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """

        # is there an obstacle in front of us?
        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            if debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        # check for the state of the traffic lights
        light_state, traffic_light = self._is_light_red(lights_list)
        if light_state and self._follow_traffic_lights:
            if debug:
                print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

            self._state = AgentState.BLOCKED_RED_LIGHT
            hazard_detected = True

        if hazard_detected:
            control = self.emergency_stop()
        else:
            self._state = AgentState.NAVIGATING
            # standard local planner behavior
            control = self._local_planner.run_step()

        return control
