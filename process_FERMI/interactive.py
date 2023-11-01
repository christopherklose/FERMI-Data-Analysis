"""
Library for interactive widgets

2022/23
@authors:   MS: Michael Schneider (mschneid@mbi-berlin.de)
            CK: Christopher Klose (christopher.klose@mbi-berlin.de)
"""

import numpy as np
import h5py

import scipy as sp
from scipy.ndimage import gaussian_filter
from scipy.ndimage import fourier_shift
from ipywidgets import FloatRangeSlider, FloatSlider, Button, interact, IntSlider
from scipy.constants import c, h, e

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

import ipywidgets

import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.detectors import Detector

#from fth import reconstruct, shift_image, propagate, shift_phase


def shift_image(image,shift):
    '''
    Shifts image with sub-pixel precission via Fourier space
    
    
    Parameters
    ----------
    image: array
        Moving image, will be shifted by shift vector
        
    shift: vector
        x and y translation in px
    
    Returns
    -------
    image_shifted: array
        Shifted image
    -------
    author: CK 2021
    '''
    
    #Shift Image
    shift_image = fourier_shift(scp.fft.fft2(image,workers=-1), shift)
    shift_image = scp.fft.ifft2(shift_image,workers=-1)
    shift_image = shift_image.real

    return shift_image


#Draw circle mask
def circle_mask(shape,center,radius,sigma='none'):

    '''
    Draws circle mask with option to apply gaussian filter for smoothing
    
    Parameter
    =========
    shape : int tuple
        shape/dimension of output array
    center : int tuple
        center coordinates (ycenter,xcenter)
    radius : scalar
        radius of mask in px. Care: diameter is always (2*radius+1) px
    sigma : scalar
        std of gaussian filter
        
    Output
    ======
    mask: array
        binary mask, or smoothed binary mask        
    ======
    author: ck 2022
    '''
    
    #setup array
    x = np.linspace(0,shape[1]-1,shape[1])
    y = np.linspace(0,shape[0]-1,shape[0])
    X,Y = np.meshgrid(x,y)

    # define circle
    mask = np.sqrt(((X-center[1])**2+(Y-center[0])**2)) <= (radius)
    mask = mask.astype(float)

    # smooth aperture
    if sigma != 'none':
        mask = gaussian_filter(mask,sigma)
           
    return mask

def cimshow(im, **kwargs):
    """Simple 2d image plot with adjustable contrast.
    
    Returns matplotlib figure and axis created.
    """
    im = np.array(im)
    fig, ax = plt.subplots()
    im0 = im[0] if len(im.shape) == 3 else im
    mm = ax.imshow(im0, **kwargs)

    cmin, cmax, vmin, vmax = np.nanpercentile(im, [.1, 99.9, .001, 99.999])
    # vmin, vmax = np.nanmin(im), np.nanmax(im)
    sl_contrast = FloatRangeSlider(
        value=(cmin, cmax), min=vmin, max=vmax, step=(vmax - vmin) / 500,
        layout=ipywidgets.Layout(width='500px'),
    )

    @ipywidgets.interact(contrast=sl_contrast)
    def update(contrast):
        mm.set_clim(contrast)
    
    if len(im.shape) == 3:
        w_image = IntSlider(value=0, min=0, max=im.shape[0] - 1)
        @ipywidgets.interact(nr=w_image)
        def set_image(nr):
            mm.set_data(im[nr])
    
    
    return fig, ax


def adjust_contrast(axis):
    """Change contrast of image displayed in axis."""
    img = axis.images[-1]
    vmin, vmax = img.get_clim()
    sl_contrast = ipywidgets.FloatRangeSlider(
        value=(vmin, vmax),
        min=img.get_array().min(),
        max=img.get_array().max(),
        step=(vmax - vmin) / 500,
        layout=ipywidgets.Layout(width="500px"),
    )

    @ipywidgets.interact(contrast=sl_contrast)
    def update(contrast):
        img.set_clim(contrast)


class InteractiveCenter:
    """Plot image with controls for contrast and beamstop alignment tools."""
    def __init__(self, im, c0=None, c1=None, rBS=15, **kwargs):
        im = np.array(im)
        self.fig, self.ax = cimshow(im, **kwargs)
        self.mm = self.ax.get_images()[0]
        
        if c0 is None:
            c0 = im.shape[-2] // 2
        if c1 is None:
            c1 = im.shape[-1] // 2
        
        self.c0 = c0
        self.c1 = c1
        self.rBS = rBS
        
        self.circles = []
        for i in range(5):
            color = 'g' if i == 1 else 'r'
            circle = plt.Circle([c0, c1], 10 * (i + 1), ec=color, fill=False)
            self.circles.append(circle)
            self.ax.add_artist(circle)

        w_c0 = ipywidgets.IntText(value=c0,step = 0.5, description="c0")
        w_c1 = ipywidgets.IntText(value=c1,step = 0.5, description="c1")
        w_rBS = ipywidgets.IntText(value=rBS, description="rBS")
        
        ipywidgets.interact(self.update, c0=w_c0, c1=w_c1, r=w_rBS)
    
    def update(self, c0, c1, r):
        self.c0 = c0
        self.c1 = c1
        self.rBS = r
        for i, c in enumerate(self.circles):
            c.set_center([c1, c0])
            c.set_radius(r * (i + 1))

def axis_to_roi(axis, labels=None):
    """
    Generate numpy slice expression from bounds of matplotlib figure axis.
    
    If labels is not None, return a roi dictionary for xarray.
    """
    x0, x1 = sorted(axis.get_xlim())
    y0, y1 = sorted(axis.get_ylim())
    if labels is None:
        roi = np.s_[
            int(round(y0)):int(round(y1)),
            int(round(x0)):int(round(x1))
        ]
    else:
        roi = {
            labels[0]: slice(int(round(y0)), int(round(y1))),
            labels[1]: slice(int(round(x0)), int(round(x1)))
        }
    return roi


def intensity_scale(im1, im2, mask=None):
    mask = mask if mask is not None else 1
    diff = (im1 - im2) * mask
    fig, ax = plt.subplots()
    hist, bins, patches = ax.hist(mask.flatten(), np.linspace(-100, 100, 201))
    ax.set_yscale("log")
    ax.axvline(0, c='r', lw=.5)
    ax.grid(True)

    @ipywidgets.interact(f=(.2, 2.0, .001))
    def update(f):
        diff = mask * (im1 - f * im2)
        hist, _ = np.histogram(diff, bins)
        for p, v in zip(patches, hist):
            p.set_height(v)
    return fig, ax
    
    
    
class AzimuthalIntegrationCenter:
    """Plot image with controls for contrast and center alignment tools."""

    def __init__(self, im, ai, c0=None, c1=None, mask = None, **kwargs):
        # Get center
        self.im = np.array(im)
        if c0 is None:
            c0 = im.shape[-2] // 2
        if c1 is None:
            c1 = im.shape[-1] // 2
        
        #Variables
        self.c0 = c0
        self.c1 = c1
        self.mask = mask
        self.radial_range = kwargs["radial_range"]
        self.im_data_range = kwargs["im_data_range"]
        self.pixel_size1 = ai.detector.get_pixel1()
        self.pixel_size2 = ai.detector.get_pixel2()
        self.qlines = kwargs["qlines"]
        self.ai = ai

        # Calc azimuthal integration
        self.I_t, self.q_t, self.phi_t = self.ai.integrate2d(
            self.im,
            500,
            radial_range=self.radial_range,
            unit="q_nm^-1",
            correctSolidAngle=False,
            dummy=np.nan,
            mask = self.mask,
            method="cython",
        )
        self.mI_t = np.nanmean(self.I_t, axis=0)

        # Plot
        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 4))
        # 1d Ai
        self.ax[0].plot(self.q_t, self.mI_t)
        self.ax[0].set_xlim(self.radial_range)
        self.ax[0].set_xlabel("q in 1/nm")
        self.ax[0].set_ylabel("Mean Integrated Intensity")
        self.ax[0].grid()
        # 2d Ai
        mi, ma = np.nanpercentile(self.I_t, self.im_data_range)
        self.ax[1].imshow(self.I_t, vmin=mi, vmax=ma)
        self.ax[1].set_ylabel("Angle")
        self.ax[1].grid()

        # qlines
        for qt in self.qlines:
            self.ax[1].axvline(qt, ymin=0, ymax=360, c="red")

        w_c0 = ipywidgets.FloatSlider(value=c0,min=im.shape[-2]/2-np.round(im.shape[-2]/6),max=3/2*im.shape[-2],step=5, description="y-center",layout=ipywidgets.Layout(width="500px"))
        w_c1 = ipywidgets.FloatSlider(value=c1,min=im.shape[-1]/2-np.round(im.shape[-1]/6),max=3/2*im.shape[-1],step=5, description="x-center",layout=ipywidgets.Layout(width="500px"))

        ipywidgets.interact(self.update, c0=w_c0, c1=w_c1)

    def update(self, c0, c1, **kwargs):
        self.c0 = c0
        self.c1 = c1

        self.ai.poni1 = (
            self.c0 * self.pixel_size1)  # y (vertical)
        self.ai.poni2 = (
            self.c1 * self.pixel_size2)  # x (horizontal)

        self.I_t, self.q_t, self.phi_t = self.ai.integrate2d(
            self.im,
            500,
            radial_range=self.radial_range,
            unit="q_nm^-1",
            correctSolidAngle=False,
            dummy=np.nan,
            mask = self.mask,
            #method="cython"
        )
        self.mI_t = np.nanmean(self.I_t, axis=0)

        # Plot
        # 1d Ai
        self.ax[0].clear()
        self.ax[0].plot(self.q_t, self.mI_t)
        self.ax[0].set_xlabel("q in 1/nm")
        self.ax[0].set_ylabel("Mean Integrated Intensity")
        self.ax[0].grid()

        # 2d Ai
        mi, ma = np.nanpercentile(self.I_t, self.im_data_range)
        self.ax[1].imshow(self.I_t, vmin=mi, vmax=ma)
        
        
class InteractiveCircleCoordinates:
    """For drawing of circles, i.e., phase retrieval mask."""

    def __init__(self, im, nr_circ, coordinates="None", **kwargs):
        # Display image
        im = np.array(im)
        self.fig, self.ax = cimshow(im, **kwargs)
        self.mm = self.ax.get_images()[0]

        # Create list of aperture coordinates
        if coordinates == "None":
            self.c_yxr = []
            for i in range(nr_circ):
                self.c_yxr.append([im.shape[-2] // 2, im.shape[-1] // 2, 15])
        else:
            self.c_yxr = coordinates

        # Draw circles
        self.circles = []
        for i in range(nr_circ):
            color = "r" if i == 0 else "g"
            circle = plt.Circle(
                [self.c_yxr[i][1], self.c_yxr[i][0]],
                self.c_yxr[i][2],
                ec=color,
                fill=False,
            )
            self.circles.append(circle)
            self.ax.add_artist(circle)

        # Create slider to select aperture
        w_idx = ipywidgets.IntSlider(
            value=0,
            min=0,
            max=nr_circ - 1,
            step=1,
            description="circle index",
            layout=ipywidgets.Layout(width="250px"),
        )

        # Create circle widget sliders
        label = ["yc", "xc", "r0"]
        width = ["600px", "600px", "500px"]
        mi = [0, 0, 0]
        ma = [2*im.shape[-2], 2*im.shape[-1], 2*np.max([im.shape[-2],im.shape[-1]])]
        w = [
            ipywidgets.FloatSlider(
                value=self.c_yxr[0][k],
                min=mi[k],
                max=ma[k],
                step=0.5,
                description=label[k],
                layout=ipywidgets.Layout(width=width[k]),
            )
            for k in range(3)
        ]
        self.w = w

        # Interactive user interface
        iidx = ipywidgets.interact(self.update_index, idx_circ=w_idx)
        icirc = ipywidgets.interact(
            self.update,
            c0=self.w[0],
            c1=self.w[1],
            r=self.w[2],
        )

    # Update initial widget values to the ones from list when changing between circles
    def update_index(self, idx_circ):
        self.idx = idx_circ
        cy = self.c_yxr[self.idx][0]
        cx = self.c_yxr[self.idx][1]
        cr = self.c_yxr[self.idx][2]

        # Keep these separated as this prevents overwriting of the individual widget values
        self.w[0].value = cy
        self.w[1].value = cx
        self.w[2].value = cr
        
        #Change color, active circle is red
        for i, c in enumerate(self.circles):
            color = "r" if i == self.idx else "g"
            c.set_edgecolor(color)

    # Update circle values
    def update(self, c0, c1, r):
        # Update coordinate dictionary
        self.c_yxr[self.idx] = [c0, c1, r]

        # Update drawn circles
        c = self.circles[self.idx]
        c.set_center([c1, c0])
        c.set_radius(r)

        print("Aperture Coordinates:")
        print(self.c_yxr)
        
        
class InteractiveBeamstop:
    """Plot image with controls for contrast and beamstop alignment tools."""
    def __init__(self, im, c0=None, c1=None, rBS=60,stdBS=4, **kwargs):        
        #Parameter coordinates
        if c0 is None:
            c0 = im.shape[-2] // 2
        if c1 is None:
            c1 = im.shape[-1] // 2
        self.center = [c0,c1]
        
        #Beamstop parameter
        self.rBS = rBS
        self.stdBS = stdBS
        
        # Create beamstop mask
        im = np.array(im)
        self.im = im
        self.mask_bs = 1 - circle_mask(
            im.shape, self.center, self.rBS, sigma = self.stdBS
        )
        self.image = np.array(im*self.mask_bs)
        
        #Plotting
        fig, ax = plt.subplots()
        self.mm = ax.imshow(self.image)
        cmin, cmax, vmin, vmax = np.nanpercentile(im, [.1, 99, .1, 99.9])
        sl_contrast = FloatRangeSlider(
        value=(cmin, cmax), min=vmin, max=vmax, step=(vmax - vmin) / 500,
        layout=ipywidgets.Layout(width='500px'),
        )
        cim = ipywidgets.interact(self.update_plt, contrast = sl_contrast)
        
        #Change beamstop parameter
        w_rBS = ipywidgets.IntText(value=self.rBS, description="rBS")
        w_std = ipywidgets.IntText(value=self.stdBS, description="stdBS")
        ipywidgets.interact(self.update_bs, r=w_rBS,std = w_std)
    
    #Update plot
    def update_plt(self,contrast):
        self.mm.set_clim(contrast)
    
    #Update bs
    def update_bs(self, r,std):
        self.rBS = r
        self.stdBS = std
        self.mask_bs = 1 - circle_mask(
            self.mask_bs.shape, self.center, r, sigma = std
        )
        self.image = self.im*self.mask_bs
        self.mm.set_data(self.image)
        
        
class draw_polygon_mask:
    """Interactive drawing of polygon masks"""

    def __init__(self, image):
        self.image = image
        self.image_plot = image
        self.full_mask = np.zeros(image.shape)
        self.coordinates = []
        self.masks = []
        self._create_widgets()
        self.draw_gui()

    def _create_widgets(self):
        self.button_add = ipywidgets.Button(
            description="Add mask",
            button_style="warning",
            layout=ipywidgets.Layout(height="auto", width="100px"),
        )
        self.button_add.on_click(self.add_mask)
        
        
        self.button_del = ipywidgets.Button(
            description="Delete mask",
            #button_style="warning",
            layout=ipywidgets.Layout(height="auto", width="100px"),
        )
        self.button_del.on_click(self.del_mask)

    def draw_gui(self):
        """Create plot and control widgets"""

        # self.fig, self.ax = cimshow(self.image)
        # self.overlay = self.ax.imshow(self.full_mask, alpha=0.2)

        # Plotting
        fig, self.ax = plt.subplots(figsize=(8,8))
        self.mm = self.ax.imshow(self.image_plot)
        # self.overlay = self.ax.imshow(self.full_mask, alpha=0.2)
        cmin, cmax, vmin, vmax = np.nanpercentile(self.image, [0.1, 99, 0.1, 99.9])

        sl_contrast = FloatRangeSlider(
            value=(cmin, cmax),
            min=vmin,
            max=vmax,
            step=(vmax - vmin) / 500,
            layout=ipywidgets.Layout(width="500px"),
        )
        cim = ipywidgets.interact(self.update_plt, contrast=sl_contrast)

        print("Click on the figure to create a polygon.")
        print("Press the 'esc' key to start a new polygon.")
        print("Try holding the 'shift' key to move all of the vertices.")
        print("Try holding the 'ctrl' key to move a single vertex.")

        self.reset_polygon_selector()
        self.output = ipywidgets.Output()
        display(self.button_add,self.button_del, self.output)

    # Update plot
    def update_plt(self, contrast):
        self.mm.set_clim(contrast)

    def reset_polygon_selector(self):
        self.selector = PolygonSelector(
            self.ax,
            lambda *args: None,
            props=dict(color="r", linestyle="-", linewidth=2, alpha=0.9),
        )

    def create_polygon_mask(self, shape, coordinates):
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        x, y = x.flatten(), y.flatten()

        points = np.vstack((x, y)).T

        path = Path(coordinates)
        mask = path.contains_points(points)
        mask = mask.reshape(shape)
        self.masks.append(mask)
        self.coordinates.append(coordinates)
        

    def combine_masks(self):
        if len(self.masks) == 0:
            self.full_mask = np.zeros(self.image.shape)
        if len(self.masks) == 1:
            self.full_mask = self.masks[0]
        elif len(self.masks) > 1:
            self.full_mask = np.sum(np.array(self.masks).astype(int), axis=0)

        self.full_mask[self.full_mask > 1] = 1

    def add_mask(self, change):
        self.create_polygon_mask(self.image.shape, self.selector.verts)
        self.combine_masks()
        self.image_plot = self.image * (1 - self.full_mask)
        self.mm.set_data(self.image_plot)
        
    def del_mask(self,change):
        self.coordinates.pop()
        self.masks.pop()
        self.combine_masks()
        self.image_plot = self.image * (1 - self.full_mask)
        self.mm.set_data(self.image_plot)