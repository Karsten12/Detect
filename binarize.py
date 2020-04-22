import imutils
import cv2
import numpy as np

if __name__ == "__main__":

    mask_file = 'images/mask_template.png'
    sample_file = 'images/sample_image.png'

    # Create mask
    image = cv2.imread(mask_file)
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    im_mask = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite('mask.png', im_mask)

    # Apply mask to sample image
    sample = cv2.imread(sample_file)
    sample_gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    result = cv2.bitwise_and(sample_gray, sample_gray, mask = im_mask)

    # Show mask applied to sample image
    im_1 = imutils.resize(sample_gray, width=700)
    im_2 = imutils.resize(result, width=700)
    masked_sample = np.vstack((im_1, im_2))
    cv2.imshow("Result", masked_sample)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




