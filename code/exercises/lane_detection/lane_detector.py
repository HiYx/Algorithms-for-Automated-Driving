from .camera_geometry import CameraGeometry
import numpy as np
import cv2


class LaneDetector():
    # TODO: You will probably want to add your own arguments to the constructor
    def __init__(self, model_path, cam_geom=CameraGeometry()):
        self.cg = cam_geom
        self.cut_v, self.grid = self.cg.precompute_grid()
        # grid 这里拿到了roadXYZ_roadframe_iso8855之内的网格，通过图片映射的网格
        # cut_v 是像素点，最远的像素点
        
        # TODO: Add variables for your lane segmentation deep learning model
        if torch.cuda.is_available():
            self.device = "cuda"
            self.model = torch.load(model_path).to(self.device)
        else:
            self.model = torch.load(model_path, map_location=torch.device("cpu"))
            self.device = "cpu"
        self.model.eval()

    def read_imagefile_to_array(self, filename):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def detect_from_file(self, filename):
        img_array = self.read_imagefile_to_array(filename)
        return self.detect(img_array)

    def detect(self, img_array):
        """
        Detects which pixels are part of a lane boundary.

        Parameters
        ----------
        img_array : array_like, shape (image_height,image_width)
            Image as a numpy array

        Returns:
        --------
        background, prob_left, prob_right: Each one array_like, shape (image_height,image_width)
            Probability maps. For example
            prob_left[v,u] = probability(pixel (u,v) is part of left lane boundary)
        """
        # TODO: Use your lane segmentation deep learning model to implement this function
        with torch.no_grad():
            image_tensor = img_array.transpose(2,0,1).astype('float32')/255
            x_tensor = torch.from_numpy(image_tensor).to(self.device).unsqueeze(0)
            model_output = torch.softmax(self.model.forward(x_tensor), dim=1).cpu().numpy()
        background, left, right = model_output[0,0,:,:], model_output[0,1,:,:], model_output[0,2,:,:] 
        
        return background, left, right
        

    def fit_poly(self, probs):
        """ 
        Fits a polynomial of order 3 to the lane boundary.
        
        Parameters
        ----------
        probs : array_like, shape (image_height,image_width)
            Probability for each pixel that it shows a lane boundary

        Returns:
        --------
        poly: numpy.poly1d
            numpy poly1d object representing the lane boundary as a polynomial.
            The polynomial is y(x)=c0+c1*x+c2*x**2+c3*x**3, where x is the forward
            direction along the road in meters and y is the sideways direction.
            Hint:
            You will want to use self.grid here. 
            self.grid[:,0] contains all x values
            self.grid[:,1] contains all y values
            np.ravel(probs[self.cut_v:, :]) contains all probability values.
        """
        # TODO: Implement this function. You will need self.cut_v, and self.grid 
        # ravel函数的功能是将原数组拉伸成为一维数组
        probs_flat = np.ravel(probs[self.cut_v:, :])# 这个要看推理模型输出的prob是怎么排列的,shape (image_height,image_width),就是一堆概率值
        # 我们把概率值大于0.3 的点挑出来。然后形成一个mask ，这就是一个索引地址。
        mask = probs_flat>0.3# 这里生成的是一个列表
        # 然后根据索引地址，在grid 中找到对应的三维坐标值。因为这就是一个映射关系了。
        # 对应的 x = self.grid[:,0][mask]  y=self.grid[:,1][mask]
        if mask.sum() > 0:
            coeffs = np.polyfit(self.grid[:,0][mask], self.grid[:,1][mask], deg=3, w=probs_flat[mask])
        else:
            coeffs = np.array([0.,0.,0.,0.])
        
        
        
        return np.poly1d(coeffs)

    def __call__(self, image):
        if isinstance(image, str):
            image = self.read_imagefile_to_array(image)
        left_poly, right_poly, _, _ = self.get_fit_and_probs(image)
        return left_poly, right_poly

    def get_fit_and_probs(self, img):
        _, left, right = self.detect(img)
        left_poly = self.fit_poly(left)
        right_poly = self.fit_poly(right)
        return left_poly, right_poly, left, right

    
