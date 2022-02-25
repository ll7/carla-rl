#!/usr/bin python3

import carla
import random
import time
from datetime import datetime
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)



def main():
    logging.info("main")
    logging.info("Using the following egg file: \n" + carla.__file__)
    now = datetime.now().strftime('%Y-%m-%d_%H%M')
    logging.info(now)
    
    actor_list = []
    
    desired_town = 'Town01'
    
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        
        world = client.get_world()
        map = world.get_map()
        if not map.name.endswith(desired_town):
            logging.info("We want to use {}, but the map is named {}".format(desired_town, map.name))
            world = client.load_world('Town01')
            time.sleep(5)
        logging.info("Map {} loaded".format(map.name))
            
        
        blueprint_library = world.get_blueprint_library()
        
        bp = blueprint_library.filter('0012')[0] # random.choice(blueprint_library.filter('walker'))
        
        transform = random.choice(world.get_map().get_spawn_points())
        
        walker = world.spawn_actor(bp, transform)
        
        actor_list.append(walker)
        logging.debug('created %s' % walker.type_id)
        
        # set spectator as top down view
        
        spectator = world.get_spectator()
        
        spectator_transform = walker.get_transform()
        
        spectator_transform.location.z += 20.0
        spectator_transform.rotation.pitch = -90.0
        
        spectator.set_transform(spectator_transform)
        
        # create a segmentation camera and attach to walker
        
        seg_camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        
        camera_transform = carla.Transform(carla.Location(z=2.0), carla.Rotation(pitch=-90.0))
        
        seg_camera_bp.set_attribute('image_size_x', '128')
        seg_camera_bp.set_attribute('image_size_y', '128')
        seg_camera_bp.set_attribute('fov', '170.0')
        
        camera = world.spawn_actor(seg_camera_bp, camera_transform, attach_to=walker)
        
        
        
        actor_list.append(camera)
        logging.debug('created %s' % camera.type_id)
        
        camera.listen(lambda image: image.save_to_disk('./tmp/{}/{}.png'.format(now, image.frame_number), carla.ColorConverter.CityScapesPalette))
        
        time.sleep(10)
        
    finally:
        logging.debug('finally')
        logging.debug(camera)
        camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        logging.debug(camera)
        logging.debug('done.')
        

    
    
if __name__=="__main__":
    main()