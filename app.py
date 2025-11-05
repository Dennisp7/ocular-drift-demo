import numpy as np
from pyscript import display
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# --- UI getters from DOM ---
from js import document

def _get_float(id_):
    return float(document.getElementById(id_).value)

def _get_int(id_):
    return int(float(document.getElementById(id_).value))

# --- core helpers ---
def make_1_over_f_image(n=256, alpha=2.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, n))
    X = np.fft.rfft2(x)
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.rfftfreq(n)[None, :]
    f = np.sqrt(fx**2 + fy**2)
    f[0, 0] = 1e-6
    H = 1.0 / (f**(alpha/2.0))
    Y = X * H
    y = np.fft.irfft2(Y, s=(n, n))
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    return y

def radial_power_spectrum(img):
    n = img.shape[0]
    F = np.fft.fftshift(np.fft.fft2(img))
    P = np.abs(F)**2
    cy = cx = n // 2
    y, x = np.ogrid[:n, :n]
    r = np.hypot(y - cy, x - cx).astype(int)
    num = np.bincount(r.ravel(), P.ravel())
    den = np.bincount(r.ravel())
    radial_mean = num / (den + 1e-12)
    freqs = np.arange(len(radial_mean)) / (n/2.0)
    return freqs, radial_mean

def center_surround(img, sig_c=1.0, sig_s=3.0, Ac=1.0, As=0.8):
    Ic = gaussian_filter(img, sig_c)
    Is = gaussian_filter(img, sig_s)
    return Ac * Ic - As * Is

def temporal_difference(img, shift=(1,0)):
    dy, dx = shift
    img2 = np.roll(img, shift=(dy, dx), axis=(0,1))
    diff = img2 - img
    return img2, diff

# --- render helpers (each draws one figure to a <canvas> by CSS selector) ---
def _to_canvas(selector, fig):
    display(fig, target=selector, append=False, clear=True)
    plt.close(fig)

def _current_params():
    n    = _get_int("n")
    a    = _get_float("alpha")
    sc   = _get_float("sigc")
    ss   = _get_float("sigs")
    dy   = _get_int("dy")
    dx   = _get_int("dx")
    return n, a, sc, ss, dy, dx

def render_inputs(img_sel, dog_sel):
    n, a, sc, ss, _, _ = _current_params()
    img = make_1_over_f_image(n=n, alpha=a, seed=0)
    dog = center_surround(img, sig_c=sc, sig_s=ss, Ac=1.0, As=0.8)

    for arr, sel, title in [(img, img_sel, "Input (1/f^α)"), (dog, dog_sel, "DoG-filtered")]:
        fig, ax = plt.subplots(figsize=(4.6,4.6))
        ax.imshow(arr, cmap="gray", vmin=arr.min(), vmax=arr.max())
        ax.set_title(title); ax.axis("off")
        _to_canvas(sel, fig)

def render_psd_inputs(psd_sel):
    n, a, sc, ss, _, _ = _current_params()
    img = make_1_over_f_image(n=n, alpha=a, seed=0)
    dog = center_surround(img, sig_c=sc, sig_s=ss, Ac=1.0, As=0.8)
    f0, P0 = radial_power_spectrum(img)
    f1, P1 = radial_power_spectrum(dog)

    fig, ax = plt.subplots(figsize=(5.6,4.8))
    ax.plot(f0+1e-9, P0+1e-12, label="Input")
    ax.plot(f1+1e-9, P1+1e-12, label="DoG")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.6)
    ax.set_title("Radial Power Spectrum (log–log)"); ax.legend()
    _to_canvas(psd_sel, fig)

def render_drift_frames(t_sel, t1_sel, diff_sel):
    n, a, _, _, dy, dx = _current_params()
    img = make_1_over_f_image(n=n, alpha=a, seed=1)
    shifted, diff = temporal_difference(img, shift=(dy, dx))
    for arr, sel, title in [
        (img, t_sel, "Frame t"),
        (shifted, t1_sel, "Frame t+1 (shifted)"),
        (diff, diff_sel, "Temporal difference")]:
        fig, ax = plt.subplots(figsize=(4.6,4.6))
        ax.imshow(arr, cmap="gray")
        ax.set_title(title); ax.axis("off")
        _to_canvas(sel, fig)

def render_psd_drift(psd_sel):
    n, a, _, _, dy, dx = _current_params()
    img = make_1_over_f_image(n=n, alpha=a, seed=1)
    _, diff = temporal_difference(img, shift=(dy, dx))
    f_in, Pin = radial_power_spectrum(img)
    f_df, Pdf = radial_power_spectrum(np.abs(diff))

    fig, ax = plt.subplots(figsize=(5.6,4.8))
    ax.plot(f_in+1e-9, Pin+1e-12, label="Input")
    ax.plot(f_df+1e-9, Pdf+1e-12, label="Temporal diff")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.6)
    ax.set_title("PSD: Input vs Temporal diff"); ax.legend()
    _to_canvas(psd_sel, fig)

def update_all():
    # Re-render all panels with the current UI values
    render_inputs("#img_canvas", "#dog_canvas")
    render_psd_inputs("#psd_inputs_canvas")
    render_drift_frames("#t_canvas", "#t1_canvas", "#diff_canvas")
    render_psd_drift("#psd_drift_canvas")
