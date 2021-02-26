import argparse
import cv2
import nnio
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Measure inference time on dummy image input'
    )
    parser.add_argument(
        '--device', type=str, default='CPU',
        required=False,
        help='Device. CPU or TPU. Set TPU:0 or TPU:1 to use specific device.')
    parser.add_argument(
        '--speed-test-iters', type=int, default=0,
        required=False,
        help='Number of iterations to test speed.')
    args = parser.parse_args()

    # Load models
    model = nnio.zoo.edgetpu.detection.SSDMobileNet(device=args.device)

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
    cv2.imwrite('results/edgetpu_ssd.png', image_rgb[:,:,::-1])

    # Measure inference time
    if args.speed_test_iters > 0:
        times = []
        start = time.time()
        for _ in range(args.speed_test_iters):
            _, info = model(image_prepared, return_info=True)
            times.append(info['invoke_time'])
            print(info)
        end = time.time()
        time_avg = sum(times) / len(times)
        time_all = (end - start) / len(times)
        print('Average inference time: {:.02f} ms'.format(time_avg * 1000))
        print('Average summary time: {:.02f} ms'.format(time_all * 1000))

        percentiles = [0, 50, 99, 100]
        results = np.percentile(times, percentiles)
        print('Percentiles:')
        for p, res in zip(percentiles, results):
            print('{}%: {:.02f} ms'.format(p, res * 1000))

if __name__ == '__main__':
    main()

