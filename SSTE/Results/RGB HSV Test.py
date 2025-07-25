import cv2
import numpy as np
import matplotlib.pyplot as plt

fake_rgb = cv2.imread("Full Pipeline.png")  # Loads in BGR format
hsv_encoded = cv2.cvtColor(fake_rgb, cv2.COLOR_BGR2RGB)   # Get back the original HSV values

real_rgb = cv2.cvtColor(hsv_encoded, cv2.COLOR_HSV2RGB)

# Plotting
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(hsv_encoded)
plt.title("HSV")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(real_rgb)
plt.title("RGB")
plt.axis('off')

plt.tight_layout()
plt.show()