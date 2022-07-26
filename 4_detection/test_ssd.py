from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys


net_type = 'mb2-ssd-lite'
model_path = 'models/mb2-ssd-lite-mp-0_686.pth'
label_path = 'models/voc-model-labels.txt'
image_path = 'airport.jpg'

class_names = [name.strip() for name in open(label_path).readlines()]

net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(model_path)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)

orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
boxes, labels, probs = predictor.predict(image, 10, 0.4)

for i in range(boxes.size(0)):
    box = boxes[i, :]
    cv2.rectangle(orig_image, (int(box[0]),int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
    #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.putText(orig_image, label, (int(box[0]) + 20, int(box[1]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

path = "run_ssd_example_output.jpg"
cv2.imwrite(path, orig_image)
print(f"Found {len(probs)} objects. The output image is {path}")

cv2.imshow('Output', orig_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


