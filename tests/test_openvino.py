import cv2
import nnio


def main():
    # Load model
    model = nnio.zoo.openvino.detection.SSDMobileNetV2()

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
    cv2.imwrite('result.png', image_rgb[:,:,::-1])

if __name__ == '__main__':
    main()
