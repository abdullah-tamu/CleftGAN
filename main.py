import argparse
import os
import pickle
import cv2
import numpy as np
import torch


def correct_img_vals(img):
    img[img >= 1] = 1
    img[img <= -1] = -1
    return img


device = 'cuda:0'


def run(nimg, res_path, show_images):
    with open('network-snapshot-000460_stylegan3t.pkl', 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

    for i in range(1, nimg):
        z = torch.randn([1, G.z_dim]).cuda()  # latent codes
        c = None  # class labels
        img = G(z, c, truncation_psi=0.99)
        img = correct_img_vals(img)

        nimg = img[0].permute(1, 2, 0).detach().cpu().numpy()
        nimg = nimg * 127.5 + 128
        dst = np.asarray(cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        dst2 = np.asarray(cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB), dtype=np.uint8)

        x = 400
        y = 250
        w = 450
        h = 500
        dst2 = dst2[x:x + w, y:y + h]
        dst = dst[x:x + w, y:y + h]
        cv2.putText(dst2, 'press s to save the image, q to exit,', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 2)

        cv2.putText(dst2, 'or any key to skip saving the image', (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 2)

        if show_images:
            cv2.imshow('s', dst2)
            k = cv2.waitKey()
            if k == 115:
                cv2.imwrite(res_path + '/image_' + str(i) + '.png', dst)
            elif k == 113:
                quit()


def main():
    parser_main = argparse.ArgumentParser(add_help=False, description='Face Normalizer',
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_main.add_argument('--nimg', default=100, type=int)
    parser_main.add_argument('--output_path', default=f'./faces', type=str)
    parser_main.add_argument('--show_images', default=True, type=bool)

    args_main, _ = parser_main.parse_known_args()

    nimg = args_main.nimg
    res_path = args_main.output_path
    show_images = args_main.show_images

    run(nimg, res_path, show_images)


if __name__ == '__main__':
    main()
