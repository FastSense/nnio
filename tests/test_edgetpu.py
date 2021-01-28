import argparse
import cv2
import nnio


def main():
    parser = argparse.ArgumentParser(
        description='Measure inference time on dummy image input'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        required=False,
        help='Device. Set :0 to use EdgeTPU. If not specified, will use CPU.')
    args = parser.parse_args()

    # Load models
    model = nnio.zoo.edgetpu.detection.SSDMobileNet(device=args.device)

    # Get preprocessing function
    preproc = model.get_preprocessing()

    # Read image
    image_rgb = cv2.imread('dogs.jpg')[:,:,::-1].copy()

    # Pass to the neural network
    image_prepared = preproc('dogs.jpg')
    boxes = model(image_prepared)

    # Draw boxes
    for box in boxes:
        print(box)
        image_rgb = box.draw(image_rgb)

    # Write result
    cv2.imwrite('results/openvino_ssd.png', image_rgb[:,:,::-1])

if __name__ == '__main__':
    main()

