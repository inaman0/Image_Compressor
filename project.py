import numpy as np
import cv2
import pywt
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

def apply_dwt_lossless(channel: np.ndarray) -> np.ndarray:
    h, w = channel.shape
    coeffs = pywt.wavedec2(channel.astype(np.float32), wavelet='haar', level=2)
    new_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        new_coeffs.append(tuple(
            pywt.threshold(d, 0.5, mode='soft') for d in detail))
    rec = pywt.waverec2(new_coeffs, wavelet='haar')
    return np.clip(rec[:h, :w], 0, 255).astype(np.uint8)

def apply_dct_compression(channel: np.ndarray, quality: int = 10) -> np.ndarray:
    h, w = channel.shape
    ph = ((h + 7) // 8) * 8
    pw = ((w + 7) // 8) * 8
    padded = np.zeros((ph, pw), dtype=np.float32)
    padded[:h, :w] = channel.astype(np.float32)
    result = np.zeros_like(padded)
    q_factor = float(max(1, 100 - quality))
    quant = np.full((8, 8), q_factor, dtype=np.float32)
    for i in range(0, ph, 8):
        for j in range(0, pw, 8):
            blk = padded[i:i+8, j:j+8] - 128.0
            dct_blk = cv2.dct(blk)
            idct_blk = cv2.idct(np.round(dct_blk / quant) * quant) + 128.0
            result[i:i+8, j:j+8] = idct_blk
    return np.clip(result[:h, :w], 0, 255).astype(np.uint8)

def compress_image(image_bgr: np.ndarray, roi_mask: np.ndarray,
                   bg_quality: int = 10) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    mask_2d = (roi_mask[:h, :w] > 0) if roi_mask.ndim == 2 else (roi_mask[:h, :w, 0] > 0)
    result = np.zeros_like(image_bgr)
    for c in range(3):
        ch = image_bgr[:, :, c]
        dwt = apply_dwt_lossless(ch)[:h, :w]
        dct = apply_dct_compression(ch, quality=bg_quality)[:h, :w]
        result[:, :, c] = np.where(mask_2d, dwt, dct)
    return result

def auto_detect_roi(image_bgr: np.ndarray, method: str = 'canny') -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if method == 'canny':
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.4), 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        mask = cv2.dilate(edges, kernel, iterations=3)
    elif method == 'saliency':
        blurred = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        dft = np.fft.fft2(blurred)
        log_amp = np.log(np.abs(dft) + 1e-8)
        residual = log_amp.real - cv2.GaussianBlur(log_amp.real.astype(np.float32), (15, 15), 0)
        sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(dft)))) ** 2
        sal = cv2.GaussianBlur(sal.astype(np.float32), (13, 13), 0)
        sal = cv2.normalize(sal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(sal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=2)
    else:  
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2).astype(np.uint8)
        _, mask = cv2.threshold(grad, 30, 255, cv2.THRESH_BINARY)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)), iterations=2)
    return mask


