import cv2
from PIL import Image
import numpy as np
from torchvision import transforms

thresh = 130

def DoG(img,):
    # # Convert to tensor
    # img_tensor = transforms.ToTensor()(img)

    # # Convert to grayscale
    # img_gray= transforms.Grayscale()(img_tensor)

    # # Convert to numpy
    # img_gray_np = img_gray.numpy().squeeze()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = np.float32(img_gray)

    # Apply Gaussian filter

    img_DoG1 = cv2.GaussianBlur(img_gray, (5, 5), 0.4) - cv2.GaussianBlur(img_gray, (5, 5), 0.2)
    cv2.imwrite('img_DoG1.jpg', img_DoG1)
    img_DoG2 = cv2.GaussianBlur(img_gray, (5, 5), 0.8) - cv2.GaussianBlur(img_gray, (5, 5), 0.4)
    cv2.imwrite('img_DoG2.jpg', img_DoG2)
    img_DoG3 = cv2.GaussianBlur(img_gray, (5, 5), 1.6) - cv2.GaussianBlur(img_gray, (5, 5), 0.8)
    cv2.imwrite('img_DoG3.jpg', img_DoG3)

    for j in range(1,img_gray.shape[0]-1):
        for i in range(1,img_gray.shape[1]-1):
            if img_DoG2[j,i] > max(img_DoG1[j-1,i-1],img_DoG1[j-1,i],img_DoG1[j-1,i+1],
                                    img_DoG1[j,i-1],img_DoG1[j,i],img_DoG1[j,i+1],
                                    img_DoG1[j+1,i-1],img_DoG1[j+1,i],img_DoG1[j+1,i+1],
                                    img_DoG2[j-1,i-1],img_DoG2[j-1,i],img_DoG2[j-1,i+1],
                                    img_DoG2[j,i-1],img_DoG2[j,i+1],
                                    img_DoG2[j+1,i-1],img_DoG2[j+1,i],img_DoG2[j+1,i+1],
                                    img_DoG3[j-1,i-1],img_DoG3[j-1,i],img_DoG3[j-1,i+1],
                                    img_DoG3[j,i-1],img_DoG3[j,i],img_DoG3[j,i+1],
                                    img_DoG3[j+1,i-1],img_DoG3[j+1,i],img_DoG3[j+1,i+1]):
                cv2.circle(img_gray, (j,i), 5, (0), 2)

    # img_gray_np = cv2.cornerHarris(img_gray_np, 2, 3, 0.04)
    #  # Normalizing
    # dst_norm = np.empty(img_gray_np.shape, dtype=np.float32)
    # cv2.normalize(img_gray_np, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
    # # Drawing a circle around corners
    # for i in range(dst_norm.shape[0]):
    #     for j in range(dst_norm.shape[1]):
    #         if int(dst_norm[i,j]) > thresh:
    #             cv2.circle(dst_norm_scaled, (j,i), 5, (0), 2)
    # # # Convert to tensor
    # img_gray = torch.from_numpy(dst_norm_scaled)

    # # Convert to PIL image
    # img_blur = transforms.ToPILImage()(img_gray)
    return img_gray


def main():
    # Load image
    # img = Image.open("two-view matching/00000022.jpg")
    img = cv2.imread("two-view matching/00000022.jpg")

    # Apply DoG
    img2 = DoG(img)

    cv2.imwrite('img_DoG.jpg', img2)

if __name__ == '__main__':
    main()
