
import numpy as np
from PIL import Image
from VLM_demo import get_entitites,get_world_bboxs_list,get_action,get_state
import matplotlib.pyplot as plt

def get_world_mask_list(image_path):

    bbox = get_world_bboxs_list(image_path)
    entities = get_entitites(image_path, bbox)

    original_image = Image.open(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)
    for item in entities:
        mask = item['mask']
        label = item['label']
        bbox = item['bbox']

        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2


        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        plt.text(x_center, y_center, label, color="black", fontsize=16, ha='center', va='center')

        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        plt.gca().imshow(mask_image)

    plt.axis('off')
    # 保存图像到本地
    output_image_path = "tmp/mask_image.png"
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)

    plt.show()

    return output_image_path, entities

if __name__ == '__main__':
    image_path = "test.jpeg"
    output_image_path, entities = get_world_mask_list(image_path)

    action = get_action(output_image_path, "Put the toy input on the spoon")
    print(action)

    for item in action["action"]:
        state = get_state(output_image_path, item)
        print(state)


