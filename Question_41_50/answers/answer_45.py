import cv2
import numpy as np
import matplotlib.pyplot as plt

def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # Gray scale
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)

    return out

def Canny(img):

	# Gaussian filter for grayscale
	def gaussian_filter(img, K_size=3, sigma=1.3):

		if len(img.shape) == 3:
			H, W, C = img.shape
		else:
			img = np.expand_dims(img, axis=-1)
			H, W, C = img.shape

		## Zero padding
		pad = K_size // 2
		out = np.zeros([H + pad * 2, W + pad * 2, C], dtype=np.float)
		out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

		## prepare Kernel
		K = np.zeros((K_size, K_size), dtype=np.float)
		for x in range(-pad, -pad + K_size):
			for y in range(-pad, -pad + K_size):
				K[y+pad, x+pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
		K /= (sigma * np.sqrt(2 * np.pi))
		K /= K.sum()

		tmp = out.copy()

		# filtering
		for y in range(H):
			for x in range(W):
				for c in range(C):
					out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])

		out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
		out = out[..., 0]

		return out


	# sobel filter
	def sobel_filter(img, K_size=3):
		H, W = img.shape

		# Zero padding
		pad = K_size // 2
		out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
		out[pad: pad + H, pad: pad + W] = gray.copy().astype(np.float)
		tmp = out.copy()

		out_v = out.copy()
		out_h = out.copy()

		## Sobel vertical
		Kv = [[1., 2., 1.],[0., 0., 0.], [-1., -2., -1.]]
		## Sobel horizontal
		Kh = [[1., 0., -1.],[2., 0., -2.],[1., 0., -1.]]

		# filtering
		for y in range(H):
			for x in range(W):
				out_v[pad + y, pad + x] = np.sum(Kv * (tmp[y: y + K_size, x: x + K_size]))
				out_h[pad + y, pad + x] = np.sum(Kh * (tmp[y: y + K_size, x: x + K_size]))

		out_v = np.clip(out_v, 0, 255)
		out_h = np.clip(out_h, 0, 255)

		out_v = out_v[pad: pad + H, pad: pad + W].astype(np.uint8)
		out_h = out_h[pad: pad + H, pad: pad + W].astype(np.uint8)

		return out_v, out_h


	def get_edge_tan(fx, fy):
		# get edge strength
		edge = np.sqrt(np.power(fx.astype(np.float32), 2) + np.power(fy.astype(np.float32), 2))
		edge = np.clip(edge, 0, 255)

		fx = np.maximum(fx, 1e-5)
		#fx[np.abs(fx) <= 1e-5] = 1e-5

		# get edge angle
		tan = np.arctan(fy / fx)

		return edge, tan


	def angle_quantization(tan):
		angle = np.zeros_like(tan, dtype=np.uint8)
		angle[np.where((tan > -0.4142) & (tan <= 0.4142))] = 0
		angle[np.where((tan > 0.4142) & (tan < 2.4142))] = 45
		angle[np.where((tan >= 2.4142) | (tan <= -2.4142))] = 95
		angle[np.where((tan > -2.4142) & (tan <= -0.4142))] = 135

		return angle


	def non_maximum_suppression(angle, edge):
		H, W = angle.shape
		
		for y in range(H):
			for x in range(W):
					if angle[y, x] == 0:
							dx1, dy1, dx2, dy2 = -1, 0, 1, 0
					elif angle[y, x] == 45:
							dx1, dy1, dx2, dy2 = -1, 1, 1, -1
					elif angle[y, x] == 90:
							dx1, dy1, dx2, dy2 = 0, -1, 0, 1
					elif angle[y, x] == 135:
							dx1, dy1, dx2, dy2 = -1, -1, 1, 1
					if x == 0:
							dx1 = max(dx1, 0)
							dx2 = max(dx2, 0)
					if x == W-1:
							dx1 = min(dx1, 0)
							dx2 = min(dx2, 0)
					if y == 0:
							dy1 = max(dy1, 0)
							dy2 = max(dy2, 0)
					if y == H-1:
							dy1 = min(dy1, 0)
							dy2 = min(dy2, 0)
					if max(max(edge[y, x], edge[y+dy1, x+dx1]), edge[y+dy2, x+dx2]) != edge[y, x]:
							edge[y, x] = 0

		return edge

	def hysterisis(edge, HT=100, LT=30):
		H, W = edge.shape

		# Histeresis threshold
		edge[edge >= HT] = 255
		edge[edge <= LT] = 0

		_edge = np.zeros((H+2, W+2), dtype=np.float32)
		_edge[1:H+1, 1:W+1] = edge

		## 8 - Nearest neighbor
		nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)

		for y in range(1, H+2):
				for x in range(1, W+2):
						if _edge[y, x] < LT or _edge[y, x] > HT:
								continue
						if np.max(_edge[y-1:y+2, x-1:x+2] * nn) >= HT:
								_edge[y, x] = 255
						else:
								_edge[y, x] = 0

		edge = _edge[1:H+1, 1:W+1]
								
		return edge

	# grayscale
	gray = BGR2GRAY(img)

	# gaussian filtering
	gaussian = gaussian_filter(gray, K_size=5, sigma=1.4)

	# sobel filtering
	fy, fx = sobel_filter(gaussian, K_size=3)

	# get edge strength, angle
	edge, tan = get_edge_tan(fx, fy)

	# angle quantization
	angle = angle_quantization(tan)

	# non maximum suppression
	edge = non_maximum_suppression(angle, edge)

	# hysterisis threshold
	out = hysterisis(edge)

	return out


def Hough_Line_step2(edge):
	## Voting
	def voting(edge):
		H, W = edge.shape
		
		drho = 1
		dtheta = 1

		# get rho max length
		rho_max = np.ceil(np.sqrt(H ** 2 + W ** 2)).astype(np.int)

		# hough table
		hough = np.zeros((rho_max, 180), dtype=np.int)

		# get index of edge
		ind = np.where(edge == 255)

		## hough transformation
		for y, x in zip(ind[0], ind[1]):
				for theta in range(0, 180, dtheta):
						# get polar coordinat4s
						t = np.pi / 180 * theta
						rho = int(x * np.cos(t) + y * np.sin(t))

						# vote
						hough[rho, theta] += 1
							
		out = hough.astype(np.uint8)

		return out

	# non maximum suppression
	def non_maximum_suppression(hough):
		rho_max, _ = hough.shape

		## non maximum suppression
		for y in range(rho_max):
			for x in range(180):
				# get 8 nearest neighbor
				x1 = max(x-1, 0)
				x2 = min(x+2, 180)
				y1 = max(y-1, 0)
				y2 = min(y+2, rho_max-1)
				if np.max(hough[y1:y2, x1:x2]) == hough[y,x] and hough[y, x] != 0:
					pass
					#hough[y,x] = 255
				else:
					hough[y,x] = 0

		# for hough visualization
		# get top-10 x index of hough table
		ind_x = np.argsort(hough.ravel())[::-1][:10]
		# get y index
		ind_y = ind_x.copy()
		thetas = ind_x % 180
		rhos = ind_y // 180
		_hough = np.zeros_like(hough, dtype=np.int)
		_hough[rhos, thetas] = 255

		return _hough

	# voting
	hough = voting(edge)

	# non maximum suppression
	out = non_maximum_suppression(hough)

	return out


# Read image
img = cv2.imread("../thorino.jpg").astype(np.float32)
gray = BGR2GRAY(img)
# Canny
#edge = Canny(img)
edge = cv2.Canny(gray, threshold1=30., threshold2=100., apertureSize=3, L2gradient=True)

# Hough
out = Hough_Line_step2(edge)

out = out.astype(np.uint8)

# Save result
cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
