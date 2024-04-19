import io
from io import BytesIO
import skimage
import torch
import numpy as np
from flask import Flask, request, jsonify
import base64
from skimage import color, io
from skimage.transform import resize
from model_lgsn import Generator_S2F  # Assuming model_lgsn.py contains the model definition
import matplotlib as plt
from flask_cors import CORS
from PIL import Image
from flask import send_file



app = Flask(__name__)
CORS(app)

def process_image(image_bytes):
    """
    Processes an image using the shadow removal model and returns the base64 encoded output.

    Args:
        image_bytes: Byte array representing the image data.

    Returns:
        str: Base64 encoded string of the processed image data, or None on error.
    """

    try:
        
        generator_A2B = 'netG_A2B.pth'
        device=torch.device('cpu')
        
        model = Generator_S2F()
        model.load_state_dict(torch.load(generator_A2B, map_location=torch.device('cpu')))
        model.eval()

        # Load image
        image = io.imread(BytesIO(image_bytes))

        # # Assuming image_bytes is a byte array
        # image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        # image = skimage.io.imread(image_np)


        # Convert to Lab color space
        labimage = color.rgb2lab(image)

        labimage448=resize(labimage,(448,448,3))
        labimage_L448=labimage448[:,:,0]
        labimage448[:,:,0]=np.asarray(labimage448[:,:,0])/50.0-1.0
        labimage448[:,:,1:]=2.0*(np.asarray(labimage448[:,:,1:])+128.0)/255.0-1.0
        labimage448=torch.from_numpy(labimage448).float()
        labimage_L448=labimage448[:,:,0]
        labimage448=labimage448.view(448,448,3)
        labimage_L448=labimage_L448.view(448,448,1)
        labimage448=labimage448.transpose(0, 1).transpose(0, 2).contiguous()
        labimage448=labimage448.unsqueeze(0).to(device)
        labimage_L448=labimage_L448.transpose(0, 1).transpose(0, 2).contiguous()
        labimage_L448=labimage_L448.unsqueeze(0).to(device)

        labimage480=resize(labimage,(480,640,3))
        labimage_L480=labimage480[:,:,0]
        labimage480[:,:,0]=np.asarray(labimage480[:,:,0])/50.0-1.0
        labimage480[:,:,1:]=2.0*(np.asarray(labimage480[:,:,1:])+128.0)/255.0-1.0
        labimage480=torch.from_numpy(labimage480).float()
        labimage_L480=labimage480[:,:,0]
        labimage480=labimage480.view(480,640,3)
        labimage_L480=labimage_L480.view(480,640,1)
        labimage480=labimage480.transpose(0, 1).transpose(0, 2).contiguous()
        labimage480=labimage480.unsqueeze(0).to(device)
        labimage_L480=labimage_L480.transpose(0, 1).transpose(0, 2).contiguous()
        labimage_L480=labimage_L480.unsqueeze(0).to(device)
        
        # Generate output
        temp_B448,_ = model(labimage448,labimage_L448)
        temp_B480,_ = model(labimage480,labimage_L480)

        fake_B448 = temp_B448.data
        # fake_B448[:,0]=50.0*(fake_B448[:,0]+1.0)
        fake_B448[:,1:]=255.0*(fake_B448[:,1:]+1.0)/2.0-128.0
        fake_B448=fake_B448.data.squeeze(0).cpu()
        fake_B448=fake_B448.transpose(0, 2).transpose(0, 1).contiguous().numpy()
        fake_B448=resize(fake_B448,(480,640,3))

        fake_B480 = temp_B480.data
        fake_B480[:,0]=50.0*(fake_B480[:,0]+1.0)
        # fake_B480[:,1:]=255.0*(fake_B480[:,1:]+1.0)/2.0-128.0
        fake_B480=fake_B480.data.squeeze(0).cpu()
        fake_B480=fake_B480.transpose(0, 2).transpose(0, 1).contiguous().numpy()
        fake_B480=resize(fake_B480,(480,640,3))

        fake_B=fake_B480
        fake_B[:,:,1:]=fake_B448[:,:,1:]
        outputimage=color.lab2rgb(fake_B)
        outputimage_uint8 = (outputimage * 255).astype(np.uint8)
        
        output_image = Image.fromarray(outputimage_uint8)  # Convert to PIL Image if not already
        output_image.save("output_image.jpg")  # Save the image to a file
        
        
        
        
        
        

        # Encode to base64 string
        return base64.b64encode(outputimage_uint8.tobytes()).decode('utf-8')

    except Exception as e:
        print(f"Error processing image: {e}")
        return None  # Return None on error

@app.route('/process_image', methods=['POST'])
def process_image_from_base64():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({'error': 'Missing base64_image in request data'}), 400

    image_bytes = base64.b64decode(data['image'])
    
    processed_image = process_image(image_bytes)
    
    
    
    

    if processed_image is None:
        return jsonify({'error': 'Error processing image'}), 500

    decoded_data = base64.b64decode(processed_image)
    # print(decoded_data[:500])
    image_path='output_image.jpg'
    return send_file(image_path, mimetype='image/jpeg', as_attachment=False), 200


if __name__ == '__main__':
    print("App running at localhost:5000")
    app.run(host='localhost', port=5000 , debug=True)