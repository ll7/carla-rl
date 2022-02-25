#!/usr/bin python

import carla
import random
import time

def main():
    print("main")
    print("Using the following egg file: \n" + carla.__file__)
    
    actor_list = []
    
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        
        world = client.get_world()
        
        blueprint_library = world.get_blueprint_library()
        
        bp = random.choice(blueprint_library.filter('vehicle'))
        
        transform = random.choice(world.get_map().get_spawn_points())
        
        vehicle = world.spawn_actor(bp, transform)
        
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)
        
        spectator = world.get_spectator()
        
        # vehicle_transform = vehicle.get_transform()
        
        spectator.set_transform(vehicle.get_transform())
        
        time.sleep(10)
        
    finally:
        
        
    
    
if __name__=="__main__":
    main()