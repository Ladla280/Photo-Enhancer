from flask import Flask, request, send_file
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/enhance', methods=['POST'])
def enhance():
    file = request.files['photo']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    enhanced_img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    cv2.imwrite('enhanced.jpg', enhanced_img)
    return send_file('enhanced.jpg', mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()
