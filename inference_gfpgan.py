import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from pathlib import Path
from gfpgan import GFPGANer
import time
import datetime


def main():
    """Inference demo for GFPGAN (for users).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='inputs/whole_imgs',
        help='Input image or folder. Default: inputs/whole_imgs')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder. Default: results')
    # we use version to select models, which is more user-friendly
    parser.add_argument(
        '-v', '--version', type=str, default='1.3', help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
    parser.add_argument(
        '-s', '--upscale', type=int, default=2, help='The final upsampling scale of the image. Default: 2')

    parser.add_argument(
        '--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
    parser.add_argument(
        '--bg_tile',
        type=int,
        default=400,
        help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
    parser.add_argument('-w', '--weight', type=float, default=0.5, help='Adjustable weights.')
    parser.add_argument('-r', '--sizelimit', type=int, default=2048, help='Size Limit for each photo. Default: 2048')
    parser.add_argument('--cmp', action='store_true', help='Store input and output faces for comparison.')
    parser.add_argument('--bgimg', action='store_true', help='Store upscaled background image.')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength for Real-ESRGAN General x4v3 model. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    args = parser.parse_args()

    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    if args.input.endswith('/'):
        args.input = args.input[:-1]
    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, '*')))

    os.makedirs(args.output, exist_ok=True)

    # ------------------------ set up background upsampler ------------------------
    if args.bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            from realesrgan import RealESRGANer
            from basicsr.utils.download_util import load_file_from_url
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')

            model_name = 'realesr-general-x4v3'
            model_path = os.path.join('realesrgan', model_name + '.pth')
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]
            if not os.path.isfile(model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in file_url:
                    # model_path will be updated
                    model_path = load_file_from_url(
                        url=url, model_dir=os.path.join(ROOT_DIR, 'realesrgan'), progress=True, file_name=None)

            dni_weight = None
            if args.denoise_strength != 1:
                wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
                model_path = [model_path, wdn_model_path]
                dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

            bg_upsampler = RealESRGANer(
                scale=4,
                model_path=model_path,
                dni_weight=dni_weight,
                model=model,
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # ------------------------ set up GFPGAN restorer ------------------------
    if args.version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif args.version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    elif args.version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    elif args.version == '1.4':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif args.version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    else:
        raise ValueError(f'Wrong model version {args.version}.')

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=args.upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    # ------------------------ restore ------------------------
    img_count = 0
    processed_count = 0
    t0 = time.time()
    for img_path in img_list:
        t1 = time.time()
        str_rem_time = ""
        if processed_count > 0:
            rem_time = (t1-t0) / processed_count * (len(img_list) - img_count)
            str_rem_time = str(datetime.timedelta(seconds=int(rem_time)))
        img_count += 1
        # read image
        img_name = os.path.basename(img_path)
        out_file = Path(os.path.join(args.output, 'restored_imgs', img_name))
        if out_file.is_file():
            continue
        processed_count += 1
        print(f'Processing: {img_name} [{img_count}/{len(img_list)}] ... Remaining: {str_rem_time}')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        scale_percent = 1.0
        if (input_img.shape[1] >= input_img.shape[0] and input_img.shape[1] > args.sizelimit):
            scale_percent = args.sizelimit / input_img.shape[1]
        elif (input_img.shape[0] >= input_img.shape[1] and input_img.shape[0] > args.sizelimit):
            scale_percent = args.sizelimit / input_img.shape[0]
        width = int(input_img.shape[1] * scale_percent)
        height = int(input_img.shape[0] * scale_percent)
        dim = (width, height)
        input_img = cv2.resize(input_img, dim, interpolation = cv2.INTER_AREA)

        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img, bg_img = restorer.enhance(
            input_img,
            has_aligned=args.aligned,
            only_center_face=args.only_center_face,
            paste_back=True,
            weight=args.weight)

        # save faces
        if args.cmp:
            for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
                # save cropped face
                save_crop_path = os.path.join(args.output, 'cropped_faces', f'{basename}_{idx:02d}.png')
                imwrite(cropped_face, save_crop_path)
                # save restored face
                if args.suffix is not None:
                    save_face_name = f'{basename}_{idx:02d}_{args.suffix}.png'
                else:
                    save_face_name = f'{basename}_{idx:02d}.png'
                save_restore_path = os.path.join(args.output, 'restored_faces', save_face_name)
                imwrite(restored_face, save_restore_path)
                # save comparison image
                cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                imwrite(cmp_img, os.path.join(args.output, 'cmp', f'{basename}_{idx:02d}.png'))

        # save restored img
        if restored_img is not None:
            if args.ext == 'auto':
                extension = ext[1:]
            else:
                extension = args.ext

            if args.suffix is not None:
                save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}_{args.suffix}.{extension}')
            else:
                save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}.{extension}')
            imwrite(restored_img, save_restore_path)

        if args.bgimg and bg_img is not None:
            extension = ext[1:]
            save_restore_path = os.path.join(args.output, 'upscaled_imgs', f'{basename}.{extension}')
            imwrite(bg_img, save_restore_path)

    print(f'Results are in the [{args.output}] folder.')


if __name__ == '__main__':
    main()
