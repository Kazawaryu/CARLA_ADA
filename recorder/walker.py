import copy
import csv
import os
import carla
import math

from recorder.actor import Actor
class Walker(Actor):
        def __init__(self,
                 uid,
                 name: str,
                 base_save_dir: str,
                 carla_actor: carla.Walker):
            super().__init__(uid=uid, name=name, parent=None,carla_actor=carla_actor)
            self.walker_type = copy.deepcopy(carla_actor.type_id)
            self.save_dir = '{}/{}_{}'.format(base_save_dir, self.walker_type, self.get_uid())
            self.first_tick = True
            self.auto_pilot = True
            self.vehicle_agent = None
            self.carla_actor = carla_actor

        def get_type_id(self):
            return 'others.walker'
        
        def get_save_dir(self):
            return self.save_dir
        
        def get_carla_bbox(self):
            return self.carla_actor.bounding_box
        
        def get_carla_transform(self):
            return self.carla_actor.get_transform()
        
        def get_control(self):
            """
            Get control command.
            :return: control command.
            """
            return self.carla_actor.get_control()
        
        def get_carla_actor(self):
            return self.carla_actor
            