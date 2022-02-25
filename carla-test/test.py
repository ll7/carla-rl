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
        
        # set spectator as top down view
        
        spectator = world.get_spectator()
        
        spectator_transform = vehicle.get_transform()
        
        spectator_transform.location.z += 20.0
        spectator_transform.rotation.pitch = -90.0
        
        spectator.set_transform(spectator_transform)
        
        time.sleep(10)
        
    finally:
        print('finally')    
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')

    
    
if __name__=="__main__":
    main()