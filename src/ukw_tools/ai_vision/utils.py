import cv2
import numpy as np

def apply_gradcam_on_image(img, cam, targets):
    '''img: input Tensor with dim = (3,x,y)'''
    img = img.clone().detach()
    input_tensor = img.unsqueeze(0)
    grayscale_cam = cam(input_tensor=input_tensor,targets = targets, aug_smooth = True)[0]
    grayscale_cam = grayscale_cam.T
    return grayscale_cam



inset_dim = (300,300)
vertical_space = 40
horizontal_space = 6
main_img_dim = (1000,1000)
array_template = np.zeros((1080,1920,3))

subplot_coordinates = {
    "appendix": [inset_dim[0] * 0, inset_dim[0] * 1, inset_dim[1] * 0, inset_dim[1] * 1],
    "ileocaecalvalve": [
        inset_dim[0] * 1 + vertical_space,
        inset_dim[0] * 2 + vertical_space,
        inset_dim[1] * 0,
        inset_dim[1] * 1,
    ],
    "ileum": [
        inset_dim[0] * 2 + 2 * vertical_space,
        inset_dim[0] * 3 + 2 * vertical_space,
        inset_dim[1] * 0,
        inset_dim[1] * 1,
    ],
    "main": [
        40 + main_img_dim[0] * 0,
        40 + main_img_dim[0] * 1,
        inset_dim[1] * 1 + horizontal_space,
        inset_dim[1] * 1 + horizontal_space + main_img_dim[1],
    ],
    "polyp": [
        inset_dim[0] * 0,
        inset_dim[0] * 1,
        inset_dim[1] * 1 + horizontal_space * 2 + main_img_dim[1],
        inset_dim[1] * 2 + horizontal_space * 2 + main_img_dim[1],
    ],
    "snare": [
        inset_dim[0] * 1 + vertical_space,
        inset_dim[0] * 2 + vertical_space,
        inset_dim[1] * 1 + horizontal_space * 2 + main_img_dim[1],
        inset_dim[1] * 2 + horizontal_space * 2 + main_img_dim[1],
    ],
    "wound": [
        inset_dim[0] * 2 + vertical_space * 2,
        inset_dim[0] * 3 + vertical_space * 2,
        inset_dim[1] * 1 + horizontal_space * 2 + main_img_dim[1],
        inset_dim[1] * 2 + horizontal_space * 2 + main_img_dim[1],
    ],
    "nbi": [
        inset_dim[0] * 0,
        inset_dim[0] * 1,
        inset_dim[1] * 2 + horizontal_space * 3 + main_img_dim[1],
        inset_dim[1] * 3 + horizontal_space * 3 + main_img_dim[1],
    ],
    "grasper": [
        inset_dim[0] * 1 + vertical_space,
        inset_dim[0] * 2 + vertical_space,
        inset_dim[1] * 2 + horizontal_space * 3 + main_img_dim[1],
        inset_dim[1] * 3 + horizontal_space * 3 + main_img_dim[1],
    ],
    "low_quality": [
        inset_dim[0] * 2 + vertical_space * 2,
        inset_dim[0] * 3 + vertical_space * 2,
        inset_dim[1] * 2 + horizontal_space * 3 + main_img_dim[1],
        inset_dim[1] * 3 + horizontal_space * 3 + main_img_dim[1],
    ],
}

def assemble_image(main, results):
    main = main.copy()
    main = (main-main.min())/main.max()
    inset_array = main.copy()
    inset_array = cv2.resize(inset_array, inset_dim)
    main = cv2.resize(main, main_img_dim)
    array = array_template.copy()
    # insets = {}
    for key, value in results.items():
        if not key in subplot_coordinates:
            continue
        value = cv2.resize(value, inset_dim)
        _inset_array = inset_array.copy()
        for i in range(inset_array.shape[-1]):
            _inset_array[:,:,i] = _inset_array[:,:,i]*value


        _inset_array = cv2.putText(img=_inset_array,text=key, org=(30,30), thickness = 4, color = (1,1,1), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1)
        array[subplot_coordinates[key][0] : subplot_coordinates[key][1],
                subplot_coordinates[key][2] : subplot_coordinates[key][3]] = _inset_array


    array[subplot_coordinates["main"][0] : subplot_coordinates["main"][1],
          subplot_coordinates["main"][2] : subplot_coordinates["main"][3], :] = main


    return array