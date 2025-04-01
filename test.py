{
    'Objects': ['bin', 'rubbish', 'tomato1', 'tomato2'], 
    'Query': 'pick up the rubbish and leave it in the trash can', 
    'Planner': ['grasp the rubbish', 'back to default pose', 'move to the top of the bin', 'open gripper'], 
    'Action': 'grasp the rubbish', 
    'movable': 'gripper', 
    'affordable': {
        'x': "rubbish['position'][0]", 
        'y': "rubbish['position'][1]", 
        'z': "rubbish['position'][2]", 
        'target_affordance': 1, 
        'set': 'affordance_map[x, y, z] = target_affordance'
        }, 
    'avoid': {
        'x': "bin['position'][0]", 
        'y': "bin['position'][1]", 
        'z': "bin['position'][2]", 
        'set': 'set_voxel_by_radius(avoidance_map, [x,y,z], radius_cm=5, value=1)'
        }, 
    'gripper': {
        'x': "rubbish['position'][0]", 
        'y': "rubbish['position'][1]", 
        'z': "rubbish['position'][2]", 
        'set': 'set_voxel_by_radius(gripper_map, [x,y,z], radius_cm=1, value=0)'
        }, 
    'rotation': {
        'target_rotation': "vec2quat(-rubbish['normal'])", 
        'set': 'rotation_map[:, :, :] = target_rotation'
        }, 
    'velocity': {
        'set': 'default'
        }
 }