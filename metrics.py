import numpy as np
def compute_psnr(a, b):
    
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

def compute_ssim(img1, img2):
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    scores = []
    for c in range(3):
        a, b = img1[:,:,c].astype(np.float64), img2[:,:,c].astype(np.float64)
        mu1, mu2 = np.mean(a), np.mean(b)
        scores.append(((2*mu1*mu2+c1)*(2*np.mean((a-mu1)*(b-mu2))+c2)) /
                       ((mu1**2+mu2**2+c1)*(np.var(a)+np.var(b)+c2)))
    return float(np.mean(scores))