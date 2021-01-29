import argparse
import cv2
import nnio


def main():
    parser = argparse.ArgumentParser(
        description='Measure inference time on dummy image input'
    )
    parser.add_argument(
        '--device', type=str, default='CPU',
        required=False,
        help='Device. CPU or GPU or MYRIAD.')
    args = parser.parse_args()

    # Load models
    model = nnio.zoo.openvino.detection.SSDMobileNetV2(device=args.device)

    # Get preprocessing function
    preproc = model.get_preprocessing()

    # Read image
    # pylint: disable=no-member
    image_rgb = cv2.imread('dogs.jpg')[:,:,::-1].copy()

    # Pass to the neural network
    image_prepared = preproc('dogs.jpg')
    boxes, info = model(image_prepared, return_info=True)
    print(info)

    # Draw boxes
    for box in boxes:
        print(box)
        image_rgb = box.draw(image_rgb)

    # Write result
    # pylint: disable=no-member
    cv2.imwrite('results/openvino_ssd.png', image_rgb[:,:,::-1])

if __name__ == '__main__':
    main()

