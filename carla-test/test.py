#!/usr/bin python3

import carla
import random
import time
from datetime import datetime

def main():
    print("main")
    print("Using the following egg file: \n" + carla.__file__)
    now = datetime.now().strftime('%Y-%m-%d_%H%M')
    print(now)
    
    actor_list = []
    
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        
        world = client.get_world()
        world = client.load_world('Town01')
        
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
        
        # create a segmentation camera and attach to vehicle
        
        seg_camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        
        camera_transform = carla.Transform(carla.Location(z=2.0), carla.Rotation(pitch=-90.0))
        
        seg_camera_bp.set_attribute('image_size_x', '128')
        seg_camera_bp.set_attribute('image_size_y', '128')
        seg_camera_bp.set_attribute('fov', '170.0')
        
        camera = world.spawn_actor(seg_camera_bp, camera_transform, attach_to=vehicle)
        
        
        
        actor_list.append(camera)
        print('created %s' % camera.type_id)
        
        camera.listen(lambda image: image.save_to_disk('./tmp/{}/{}.png'.format(now, image.frame_number), carla.ColorConverter.CityScapesPalette))
        
        time.sleep(10)
        
    finally:
        print('finally')
        print(camera)
        camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print(camera)
        print('done.')
        

    
    
if __name__=="__main__":
    main()