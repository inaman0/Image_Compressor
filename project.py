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

DARK    = "#0f0f14"
PANEL   = "#1a1a24"
BORDER  = "#2d2d3f"
VIOLET  = "#a78bfa"
GREEN   = "#34d399"
DIMTEXT = "#6b6b8a"
WHITE   = "#e0dfe8"
BTNBG   = "#7c3aed"
ACCBG   = "#059669"

class ROICompressionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ROI Hybrid Compression  |  DWT + DCT")
        self.root.configure(bg=DARK)
        self.root.geometry("1300x840")
        self.root.minsize(1100, 720)

        self.original_bgr  = None
        self.compressed_bgr = None
        self.roi_mask       = None
        self.image_path     = None

        self.poly_points    = []  
        self.poly_line_ids  = []  
        self.poly_dot_ids   = []   
        self.poly_preview   = None
        self.rect_start     = None
        self.rect_id        = None
        self._build_ui()

    def _build_ui(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('TFrame',    background=DARK)
        s.configure('TLabel',    background=DARK,  foreground=WHITE,  font=('Courier New', 10))
        s.configure('TButton',   background=BTNBG, foreground='white', font=('Courier New', 10, 'bold'), padding=6)
        s.map('TButton', background=[('active','#6d28d9'),('pressed','#5b21b6')])
        s.configure('Accent.TButton', background=ACCBG, foreground='white', font=('Courier New',10,'bold'), padding=6)
        s.map('Accent.TButton', background=[('active','#047857'),('pressed','#065f46')])
        # top bar
        top = tk.Frame(self.root, bg='#1a1a24', height=46)
        top.pack(fill='x')
        tk.Label(top, text="◈ ROI HYBRID COMPRESSION  [ DWT + DCT ]",
                 bg='#1a1a24', fg=VIOLET, font=('Courier New',14,'bold')).pack(side='left', padx=20, pady=10)
        tk.Label(top, text="polygon ROI  ·  lossless DWT inside  ·  DCT outside",
                 bg='#1a1a24', fg=DIMTEXT, font=('Courier New',9)).pack(side='left')
        main = tk.Frame(self.root, bg=DARK)
        main.pack(fill='both', expand=True, padx=8, pady=6)
        left = tk.Frame(main, bg=DARK, width=248)
        left.pack(side='left', fill='y', padx=(0,6))
        left.pack_propagate(False)
        self._build_left(left)
        center = tk.Frame(main, bg=DARK)
        center.pack(side='left', fill='both', expand=True)
        self._build_center(center)
        right = tk.Frame(main, bg=DARK, width=218)
        right.pack(side='right', fill='y', padx=(6,0))
        right.pack_propagate(False)
        self._build_right(right)

    def _section(self, parent, text):
        tk.Frame(parent, bg=DARK).pack(fill='x', pady=(10,0))
        tk.Label(parent, text=text, bg=DARK, fg=VIOLET,
                 font=('Courier New',10,'bold')).pack(anchor='w', padx=8)
        tk.Frame(parent, bg=BORDER, height=1).pack(fill='x', padx=8, pady=2)

    def _build_left(self, p):
        self._section(p, "① FILE")
        ttk.Button(p, text="▶  Load Image", command=self._load_image).pack(fill='x', padx=8, pady=3)
        self._section(p, "② ROI SELECTION")
        # Mode hint label
        self.mode_hint = tk.Label(p, text="", bg=DARK, fg='#6b8cff',
                                   font=('Courier New',8), wraplength=220, justify='left')
        self.mode_hint.pack(anchor='w', padx=8, pady=(0,2))
        self.roi_mode = tk.StringVar(value='polygon')
        modes = [
            ("✏  Freehand Polygon",        'polygon'),
            ("▭  Rectangle",               'rect'),
            ("⚡  Auto: Canny Edge",        'canny'),
            ("◉  Auto: Saliency",          'saliency'),
            ("∇  Auto: Gradient Threshold",'threshold'),
        ]
        for label, val in modes:
            tk.Radiobutton(p, text=label, variable=self.roi_mode, value=val,
                           bg=DARK, fg='#c4c4d4', selectcolor=BORDER,
                           activebackground=DARK, activeforeground=VIOLET,
                           font=('Courier New',9),
                           command=self._on_mode_change).pack(anchor='w', padx=12, pady=1)
        ttk.Button(p, text="⟲  Clear ROI", command=self._clear_roi).pack(fill='x', padx=8, pady=(6,2))
        ttk.Button(p, text="⚙  Detect (auto modes)", command=self._apply_auto_roi).pack(fill='x', padx=8, pady=2)
        self._section(p, "③ COMPRESSION")
        tk.Label(p, text="Background quality  (1 = max compress)",
                 bg=DARK, fg='#8888a8', font=('Courier New',8), wraplength=210).pack(anchor='w', padx=8)
        self.quality_var = tk.IntVar(value=10)
        self.qlabel = tk.Label(p, text="Quality: 10", bg=DARK, fg=GREEN,
                                font=('Courier New',10,'bold'))
        self.qlabel.pack(anchor='w', padx=8)
        tk.Scale(p, from_=1, to=50, orient='horizontal', variable=self.quality_var,
                 bg=DARK, fg=VIOLET, troughcolor=BORDER, highlightthickness=0,
                 command=lambda v: self.qlabel.config(text=f"Quality: {int(float(v))}")
                 ).pack(fill='x', padx=8)
        self._section(p, "④ RUN")
        ttk.Button(p, text="▶▶ COMPRESS", style='Accent.TButton',
                   command=self._run_compression).pack(fill='x', padx=8, pady=3)
        ttk.Button(p, text="💾  Save Result",    command=self._save_result).pack(fill='x', padx=8, pady=2)
        ttk.Button(p, text="📊  Show Difference", command=self._show_diff).pack(fill='x', padx=8, pady=2)
        self.status_var = tk.StringVar(value="Load an image to begin.")
        tk.Label(p, textvariable=self.status_var, bg=DARK, fg='#6b8cff',
                 font=('Courier New',8), wraplength=222, justify='left').pack(anchor='w', padx=8, pady=(10,4))
        
    def _build_center(self, p):
        panels = tk.Frame(p, bg=DARK)
        panels.pack(fill='both', expand=True)
        def make_panel(title, col):
            f = tk.Frame(panels, bg=PANEL)
            f.grid(row=0, column=col, padx=3, pady=3, sticky='nsew')
            panels.columnconfigure(col, weight=1)
            panels.rowconfigure(0, weight=1)
            tk.Label(f, text=title, bg=PANEL, fg=DIMTEXT,
                     font=('Courier New',9)).pack(pady=(4,0))
            return f
        lf = make_panel("ORIGINAL  [ draw ROI here ]", 0)
        rf = make_panel("COMPRESSED OUTPUT", 1)
        self.orig_canvas = tk.Canvas(lf, bg='#111118', cursor='crosshair', highlightthickness=0)
        self.orig_canvas.pack(fill='both', expand=True, padx=3, pady=3)
        self.orig_canvas.bind('<ButtonPress-1>',   self._canvas_press)
        self.orig_canvas.bind('<Double-Button-1>', self._canvas_double)
        self.orig_canvas.bind('<Motion>',          self._canvas_motion)
        self.orig_canvas.bind('<ButtonPress-3>',   self._canvas_right)  
        self.comp_canvas = tk.Canvas(rf, bg='#111118', highlightthickness=0)
        self.comp_canvas.pack(fill='both', expand=True, padx=3, pady=3)
        self.hint_bar = tk.Label(p, text="", bg='#13131e', fg=VIOLET,
                                  font=('Courier New',9), anchor='w')
        self.hint_bar.pack(fill='x', padx=3, pady=(0,2))
    def _build_right(self, p):
        tk.Label(p, text="METRICS", bg=DARK, fg=VIOLET,
                 font=('Courier New',11,'bold')).pack(anchor='w', padx=8, pady=(12,3))
        tk.Frame(p, bg=BORDER, height=1).pack(fill='x', padx=8, pady=2)
        def row(label):
            f = tk.Frame(p, bg=PANEL, pady=5)
            f.pack(fill='x', padx=8, pady=3)
            tk.Label(f, text=label, bg=PANEL, fg='#8888a8',
                     font=('Courier New',8)).pack(anchor='w', padx=8)
            v = tk.StringVar(value="—")
            tk.Label(f, textvariable=v, bg=PANEL, fg=GREEN,
                     font=('Courier New',13,'bold')).pack(anchor='w', padx=8)
            return v
        self.psnr_var     = row("PSNR (dB) — full image")
        self.psnr_roi_var = row("PSNR (dB) — ROI only")
        self.psnr_bg_var  = row("PSNR (dB) — background")
        self.ssim_var     = row("SSIM — full image")
        self.ratio_var    = row("Compression ratio")
        self.roi_pct_var  = row("ROI coverage (%)")
        self.size_var     = row("Image size")
        tk.Frame(p, bg=BORDER, height=1).pack(fill='x', padx=8, pady=8)
        tk.Label(p, text="ROI  = DWT (near-lossless)\nBG   = DCT (heavy)\n\nRight-click = undo point\nDouble-click = close poly",
                 bg=DARK, fg='#4b4b6a', font=('Courier New',8), justify='left').pack(anchor='w', padx=12)

    def _on_mode_change(self):
        mode = self.roi_mode.get()
        self._clear_roi_drawing()
        hints = {
            'polygon':   "Click to place vertices. Double-click to close. Right-click to undo.",
            'rect':      "Click and drag to draw a rectangle ROI.",
            'canny':     "Click 'Detect' to auto-detect ROI using Canny edges.",
            'saliency':  "Click 'Detect' to auto-detect salient regions.",
            'threshold': "Click 'Detect' to auto-detect via gradient threshold.",
        }
        self.mode_hint.config(text=hints.get(mode, ""))
        self.hint_bar.config(text=hints.get(mode, ""))
        cursor = 'crosshair' if mode in ('polygon','rect') else 'arrow'
        self.orig_canvas.configure(cursor=cursor)

    def _canvas_press(self, event):
        if self.original_bgr is None:
            return
        mode = self.roi_mode.get()
        if mode == 'polygon':
            self._poly_add_point(event.x, event.y)
        elif mode == 'rect':
            self.rect_start = (event.x, event.y)
            self.orig_canvas.bind('<B1-Motion>',       self._rect_drag)
            self.orig_canvas.bind('<ButtonRelease-1>', self._rect_release)

    def _canvas_double(self, event):
        if self.roi_mode.get() == 'polygon':
            self._poly_close()

    def _canvas_motion(self, event):
        if self.roi_mode.get() == 'polygon' and self.poly_points:
            if self.poly_preview:
                self.orig_canvas.delete(self.poly_preview)
            lx, ly = self.poly_points[-1]
            self.poly_preview = self.orig_canvas.create_line(
                lx, ly, event.x, event.y,
                fill='#f59e0b', width=1, dash=(4, 3))
            
    def _canvas_right(self, event):
        """Right-click: undo last polygon point."""
        if self.roi_mode.get() == 'polygon' and self.poly_points:
            self.poly_points.pop()
            if self.poly_dot_ids:
                self.orig_canvas.delete(self.poly_dot_ids.pop())
            if self.poly_line_ids:
                self.orig_canvas.delete(self.poly_line_ids.pop())

    def _poly_add_point(self, cx, cy):
        r = 4
        dot = self.orig_canvas.create_oval(
            cx-r, cy-r, cx+r, cy+r,
            fill=VIOLET, outline='white', width=1)
        self.poly_dot_ids.append(dot)
        if self.poly_points:
            lx, ly = self.poly_points[-1]
            line = self.orig_canvas.create_line(
                lx, ly, cx, cy,
                fill=VIOLET, width=2)
            self.poly_line_ids.append(line)
        self.poly_points.append((cx, cy))
        n = len(self.poly_points)
        self.hint_bar.config(
            text=f"  {n} point{'s' if n!=1 else ''} placed — double-click to close  |  right-click to undo")
        
    def _poly_close(self):
        if len(self.poly_points) < 3:
            self._set_status("Need at least 3 points to close polygon.")
            return

        x0, y0 = self.poly_points[0]
        xn, yn = self.poly_points[-1]
        close_line = self.orig_canvas.create_line(
            xn, yn, x0, y0, fill='#f59e0b', width=2, dash=(6,2))
        self.poly_line_ids.append(close_line)
        if self.poly_preview:
            self.orig_canvas.delete(self.poly_preview)
            self.poly_preview = None
        self._build_polygon_mask(self.poly_points)
        self.hint_bar.config(text="  ROI polygon closed.  Click ▶▶ COMPRESS to run.")
        
    def _build_polygon_mask(self, canvas_pts):
        if self.original_bgr is None:
            return
        h, w = self.original_bgr.shape[:2]
        cw = self.orig_canvas.winfo_width()
        ch_px = self.orig_canvas.winfo_height()
        if cw == 0 or ch_px == 0:
            return
        img_h, img_w = self.original_bgr.shape[:2]
        scale = min(cw / img_w, ch_px / img_h, 1.0)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        x_off = (cw - disp_w) // 2
        y_off = (ch_px - disp_h) // 2
        # Map canvas coords → image coords
        img_pts = []
        for (cx, cy) in canvas_pts:
            ix = int((cx - x_off) / scale)
            iy = int((cy - y_off) / scale)
            ix = max(0, min(w-1, ix))
            iy = max(0, min(h-1, iy))
            img_pts.append([ix, iy])
        pts_np = np.array(img_pts, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_np], 255)
        self.roi_mask = mask
        pct = 100.0 * np.sum(mask > 0) / mask.size
        self.roi_pct_var.set(f"{pct:.1f}%")
        self._set_status(f"Polygon ROI: {len(canvas_pts)} pts\nCoverage: {pct:.1f}%\nClick COMPRESS.")
        self._display_image_with_mask(self.orig_canvas, self.original_bgr, mask)
        self._redraw_poly_overlay()
    def _redraw_poly_overlay(self):
        """Draw the closed polygon outline on top of the mask overlay."""
        if len(self.poly_points) < 2:
            return
        pts_flat = []
        for x, y in self.poly_points:
            pts_flat.extend([x, y])
        pts_flat.extend(list(self.poly_points[0]))   # close
        self.orig_canvas.create_line(*pts_flat, fill='#f59e0b', width=2, tags='poly_outline')
        for cx, cy in self.poly_points:
            r = 3
            self.orig_canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                          fill='#f59e0b', outline='', tags='poly_outline')

    def _rect_drag(self, event):
        self.orig_canvas.delete('rect_preview')
        if self.rect_start:
            x0, y0 = self.rect_start
            self.orig_canvas.create_rectangle(
                x0, y0, event.x, event.y,
                outline=VIOLET, width=2, dash=(4,2), tag='rect_preview')
            
    def _rect_release(self, event):
        self.orig_canvas.unbind('<B1-Motion>')
        self.orig_canvas.unbind('<ButtonRelease-1>')
        self.orig_canvas.delete('rect_preview')
        if not self.rect_start:
            return
        x0c, y0c = self.rect_start
        x1c, y1c = event.x, event.y
        self.rect_start = None
        if abs(x1c-x0c) < 5 or abs(y1c-y0c) < 5:
            return
        pts = [(min(x0c,x1c), min(y0c,y1c)),
               (max(x0c,x1c), min(y0c,y1c)),
               (max(x0c,x1c), max(y0c,y1c)),
               (min(x0c,x1c), max(y0c,y1c))]
        self.poly_points = pts
        self._build_polygon_mask(pts)
        self._set_status("Rectangle ROI set.\nClick COMPRESS.")

    def _apply_auto_roi(self):
        if self.original_bgr is None:
            messagebox.showwarning("No image", "Load an image first.")
            return
        mode = self.roi_mode.get()
        if mode in ('polygon', 'rect'):
            messagebox.showinfo("Manual mode",
                                "Switch to an Auto mode first, then click Detect.")
            return
        self._set_status(f"Running {mode} detection…")
        self.root.update()
        mask = auto_detect_roi(self.original_bgr, method=mode)
        self.roi_mask = mask
        pct = 100.0 * np.sum(mask > 0) / mask.size
        self.roi_pct_var.set(f"{pct:.1f}%")
        self._display_image_with_mask(self.orig_canvas, self.original_bgr, mask)
        self._set_status(f"Auto ROI ({mode})\nCoverage: {pct:.1f}%\nClick COMPRESS.")
        self.hint_bar.config(text=f"  Auto ROI ({mode}) applied — {pct:.1f}% coverage")

    def _clear_roi(self):
        self._clear_roi_drawing()
        self.roi_mask = None
        self.roi_pct_var.set("—")
        if self.original_bgr is not None:
            self._display_image(self.orig_canvas, self.original_bgr)
        self._set_status("ROI cleared.")
        self.hint_bar.config(text="  ROI cleared.")

    def _clear_roi_drawing(self):
        for iid in self.poly_line_ids + self.poly_dot_ids:
            self.orig_canvas.delete(iid)
        self.poly_points   = []
        self.poly_line_ids = []
        self.poly_dot_ids  = []
        if self.poly_preview:
            self.orig_canvas.delete(self.poly_preview)
            self.poly_preview = None
        self.orig_canvas.delete('poly_outline')
        self.orig_canvas.delete('rect_preview')
        self.rect_start = None

    def _load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp")])
        if not path:
            return
        bgr = cv2.imread(path)
        if bgr is None:
            messagebox.showerror("Error", "Could not read image.")
            return
        self.image_path    = path
        self.original_bgr  = bgr
        self.compressed_bgr = None
        self.roi_mask      = None
        self._clear_roi_drawing()
        h, w = bgr.shape[:2]
        self._display_image(self.orig_canvas, bgr)
        self._clear_canvas(self.comp_canvas)
        self._reset_metrics()
        self.size_var.set(f"{w} × {h}")
        self._set_status(f"Loaded: {os.path.basename(path)}\n{w}×{h} px")
        self._on_mode_change()

    def _run_compression(self):
        if self.original_bgr is None:
            messagebox.showwarning("No image", "Load an image first.")
            return
        if self.roi_mask is None:
            messagebox.showwarning("No ROI", "Draw or detect an ROI first.")
            return
        self._set_status("Compressing…")
        self.root.update()
        self.compressed_bgr = compress_image(
            self.original_bgr, self.roi_mask, bg_quality=self.quality_var.get())
        self._display_image(self.comp_canvas, self.compressed_bgr)
        self._compute_metrics()
        self._set_status("Done ✓")

    def _compute_metrics(self):
        orig, comp, mask = self.original_bgr, self.compressed_bgr, self.roi_mask
        self.psnr_var.set(f"{compute_psnr(orig,comp):.2f} dB")
        self.ssim_var.set(f"{compute_ssim(orig,comp):.4f}")
        roi_px = mask > 0
        if roi_px.any():
            mse = np.mean((orig[roi_px].astype(np.float64) - comp[roi_px].astype(np.float64))**2)
            self.psnr_roi_var.set(f"{20*np.log10(255/np.sqrt(mse+1e-12)):.2f} dB")
        bg_px = mask == 0
        if bg_px.any():
            mse = np.mean((orig[bg_px].astype(np.float64) - comp[bg_px].astype(np.float64))**2)
            self.psnr_bg_var.set(f"{20*np.log10(255/np.sqrt(mse+1e-12)):.2f} dB")
        pct = 100.0 * np.sum(roi_px) / mask.size
        q   = self.quality_var.get()
        ratio = (pct/100)*1.0 + (1-pct/100)*max(1.0,(100-q)/10.0)
        self.ratio_var.set(f"~{ratio:.1f}×")
        self.roi_pct_var.set(f"{pct:.1f}%")

    def _save_result(self):
        if self.compressed_bgr is None:
            messagebox.showwarning("Nothing to save", "Run compression first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG — smaller", "*.jpg"), ("PNG — lossless", "*.png")])
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            q_bg  = max(5, self.quality_var.get())
            _, hi = cv2.imencode('.jpg', self.compressed_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            _, lo = cv2.imencode('.jpg', self.compressed_bgr, [cv2.IMWRITE_JPEG_QUALITY, q_bg])
            hi_img = cv2.imdecode(hi, cv2.IMREAD_COLOR)
            lo_img = cv2.imdecode(lo, cv2.IMREAD_COLOR)
            m3 = np.stack([self.roi_mask]*3, axis=-1) > 0
            blended = np.where(m3, hi_img, lo_img).astype(np.uint8)
            cv2.imwrite(path, blended, [cv2.IMWRITE_JPEG_QUALITY, q_bg])
        else:
            cv2.imwrite(path, self.compressed_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        orig_kb = os.path.getsize(self.image_path)/1024 if self.image_path else 0
        save_kb = os.path.getsize(path)/1024
        self._set_status(
            f"Saved: {os.path.basename(path)}\n"
            + (f"{orig_kb:.1f} KB → {save_kb:.1f} KB  ({orig_kb/max(save_kb,0.1):.1f}×)" if orig_kb else ""))
        
    def _show_diff(self):
        if self.compressed_bgr is None:
            messagebox.showwarning("No result", "Run compression first.")
            return
        diff = np.clip(cv2.absdiff(self.original_bgr, self.compressed_bgr)*5, 0, 255).astype(np.uint8)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#0f0f14')
        for ax, bgr, title in zip(axes,
            [self.original_bgr, self.compressed_bgr, diff],
            ['Original', 'Compressed', 'Difference (×5)']):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self.roi_mask is not None:
                cnts,_ = cv2.findContours(self.roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, cnts, -1, (247,159,31), 2)
            ax.imshow(rgb); ax.set_title(title, color=VIOLET, fontsize=11); ax.axis('off')
        plt.tight_layout(); plt.show()

    def _display_image(self, canvas, bgr):
        canvas.delete('all')
        self._draw_on_canvas(canvas, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def _display_image_with_mask(self, canvas, bgr, mask):
        canvas.delete('all')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
        overlay = rgb.copy()
        overlay[mask>0] = (overlay[mask>0]*0.55 + np.array([167,139,250])*0.45).astype(np.uint8)
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (167,139,250), 2)
        self._draw_on_canvas(canvas, overlay)

    def _draw_on_canvas(self, canvas, rgb):
        cw = canvas.winfo_width()  or 500
        ch = canvas.winfo_height() or 400
        h, w = rgb.shape[:2]
        scale = min(cw/w, ch/h, 1.0)
        nw, nh = int(w*scale), int(h*scale)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        tk_img = ImageTk.PhotoImage(Image.fromarray(resized))
        canvas._tk_img = tk_img
        canvas.create_image((cw-nw)//2, (ch-nh)//2, anchor='nw', image=tk_img)

    def _clear_canvas(self, canvas):
        canvas.delete('all')

    def _reset_metrics(self):
        for v in [self.psnr_var, self.psnr_roi_var, self.psnr_bg_var,
                  self.ssim_var, self.ratio_var, self.roi_pct_var]:
            v.set("—")

    def _set_status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

if __name__ == '__main__':
    root = tk.Tk()
    ROICompressionApp(root)
    root.mainloop()