import gymnasium as gym
from gymnasium import ObservationWrapper
from typing import Any
import copy
import numpy as np
import vizdoom as vzd


class TelemetryWrapper(ObservationWrapper):
    def __init__(self, env):
        self.env = env
        # how do we get the underlying game object?
        self.game = self.env.unwrapped.game
        self.depth = env.unwrapped.depth
        self.automap = env.unwrapped.automap
        self.channels = env.unwrapped.channels
        self.labels = env.unwrapped.labels
        self.telemetry = True
        self.observation_space = self.__get_observation_space()
        super().__init__(env)

    def observation(self, observation):
        new_observation = copy.deepcopy(observation)
        player_telemetry = {
            "player_x": self.game.get_game_variable(vzd.GameVariable.POSITION_X),
            "player_y": self.game.get_game_variable(vzd.GameVariable.POSITION_Y),
            "player_z": self.game.get_game_variable(vzd.GameVariable.POSITION_Z),
            "objects_in_scene": [],
            "objects_coords": [],
        }
        state = self.game.get_state()
        if state is not None:
            print("State is no longer None.")
            scene_labels = state.labels
            labels_in_scene = []
            label_coords_in_scene = []
            for label in scene_labels:
                labels_in_scene.append(label.object_name)
                label_coords_in_scene.append([label.object_position_x,
                                            label.object_position_y,
                                            label.object_position_z])
            player_telemetry["objects_in_scene"] = labels_in_scene
            player_telemetry["objects_coords"] = label_coords_in_scene
        new_observation["telemetry"] = player_telemetry
        return new_observation
    
    def __get_observation_space(self):
        """
        Returns observation space: Dict with Box entry for each activated buffer:
          "screen", "depth", "labels", "automap", "gamevariables", "telemetry"
        """
        spaces: dict[str, Any]
        spaces = {
            "screen": gym.spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                    self.channels,
                ),
                dtype=np.uint8,
            )
        }

        if self.depth:
            spaces["depth"] = gym.spaces.Box(
                0,
                255,
                (self.game.get_screen_height(), self.game.get_screen_width(), 1),
                dtype=np.uint8,
            )

        if self.labels:
            spaces["labels"] = gym.spaces.Box(
                0,
                255,
                (self.game.get_screen_height(), self.game.get_screen_width(), 1),
                dtype=np.uint8,
            )

        if self.automap:
            spaces["automap"] = gym.spaces.Box(
                0,
                255,
                (
                    self.game.get_screen_height(),
                    self.game.get_screen_width(),
                    # "automap" buffer uses same number of channels
                    # as the main screen buffer,
                    self.channels,
                ),
                dtype=np.uint8,
            )

        self.num_game_variables = self.game.get_available_game_variables_size()
        if self.num_game_variables > 0:
            spaces["gamevariables"] = gym.spaces.Box(
                np.finfo(np.float32).min,
                np.finfo(np.float32).max,
                (self.num_game_variables,),
                dtype=np.float32,
            )
        if self.telemetry:
            spaces["telemetry"] = gym.spaces.Text(
                max_length=10000,
                min_length=1
            )

        return gym.spaces.Dict(spaces)