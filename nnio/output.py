import cv2


class DetectionBox:
    def __init__(
        self,
        x_1,
        y_1,
        x_2,
        y_2,
        label=None,
        score=1.0,
    ):
        '''
        
        :parameter x_1: ``float`` in range ``[0, 1]``.
            Relative x coordinate of top-left corner.
        :parameter y_1: ``float`` in range ``[0, 1]``.
            Relative y coordinate of top-left corner.
        :parameter x_2: ``float`` in range ``[0, 1]``.
            Relative x coordinate of bottom-right corner.
        :parameter y_2: ``float`` in range ``[0, 1]``.
            Relative y coordinate of bottom-right corner.
        :parameter label: ``str`` or ``None``.
            Class label of the detected object.
        :parameter score: ``float``.
            Detection score
        '''
        self.x_1 = x_1
        self.x_2 = x_2
        self.y_1 = y_1
        self.y_2 = y_2
        self.label = label
        self.score = score

    def draw(
        self,
        image,
        color=(255,0,0),
        stroke_width=2,
        text_color=(255,0,0),
        text_width=2,
    ):
        '''
        Draws the detection box on an image

        :parameter image: numpy array.
        :parameter color: RGB color of the frame.
        :parameter stroke_width: boldness of the frame.
        :parameter text_color: RGB color of the text.
        :parameter text_width: boldness of the text.

        :return: Image with the box drawn on it.
        '''
        # Box corners:
        start_point = (
            int(image.shape[1] * self.y_1),
            int(image.shape[0] * self.x_1),
        )
        end_point = (
            int(image.shape[1] * self.y_2),
            int(image.shape[0] * self.x_2),
        )
        # Draw rectangle
        # pylint: disable=no-member
        image = cv2.rectangle(
            image,
            start_point,
            end_point,
            color, stroke_width)
        # Draw text
        if self.label is not None:
            # pylint: disable=no-member
            image = cv2.putText(
                image,
                self.label,
                (start_point[0], start_point[1] + 20 + stroke_width),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, text_color, text_width, cv2.LINE_AA
            )
        return image

    def __str__(self):
        template = 'nnio.DetectionBox(x_1={}, y_1={}, x_2={}, y_2={}, label={}, score={})'
        s = template.format(
            self.x_1, self.y_1,
            self.x_2, self.y_2,
            self.label,
            self.score)
        return s
